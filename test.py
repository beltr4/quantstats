import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import quantstatsv2 as qs

# Create sample returns data
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates, name="Strategy")

# Create a sample benchmark with lower returns
benchmark_returns = pd.Series(np.random.normal(0.0005, 0.018, len(dates)), index=dates, name="Benchmark")

def generate_plotly_charts(returns, benchmark_returns=None):
    """
    Generate interactive Plotly charts for returns data.
    Returns a dictionary of Plotly figure objects.
    """
    # Initialize dictionary to store plotly figures
    plotly_charts = {}
    
    # 1. Cumulative Returns Chart
    cum_returns = (1 + returns).cumprod()
    fig_cum = go.Figure()
    
    fig_cum.add_trace(go.Scatter(
        x=cum_returns.index,
        y=cum_returns.values,
        mode='lines',
        name=returns.name or 'Strategy',
        line=dict(color='#1f77b4', width=2)
    ))
    
    if benchmark_returns is not None:
        bench_cum_returns = (1 + benchmark_returns).cumprod()
        fig_cum.add_trace(go.Scatter(
            x=bench_cum_returns.index,
            y=bench_cum_returns.values,
            mode='lines',
            name=benchmark_returns.name or 'Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    fig_cum.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    plotly_charts['cumulative_returns'] = fig_cum
    
    # 2. Drawdown Chart
    drawdowns = qs.stats.to_drawdown_series(returns)
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=drawdowns.index,
        y=drawdowns.values * 100,  # Convert to percentage
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#d62728')
    ))
    
    fig_dd.update_layout(
        title='Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        yaxis=dict(tickformat=".2f"),
        hovermode='x unified'
    )
    
    plotly_charts['drawdown'] = fig_dd
    
    # 3. Monthly Returns Heatmap
    monthly_returns = returns.resample('M').apply(
        lambda x: (1 + x).prod() - 1
    ).to_frame()
    
    monthly_returns.columns = ['returns']
    monthly_returns['year'] = monthly_returns.index.year
    monthly_returns['month'] = monthly_returns.index.month
    
    # Pivot to get year as rows and month as columns
    pivot_returns = monthly_returns.pivot(index='year', columns='month', values='returns')
    
    # Convert to percentages
    pivot_returns = pivot_returns * 100
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_returns.columns = [month_names[i-1] for i in pivot_returns.columns]
    
    # Create the heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_returns.values,
        x=pivot_returns.columns,
        y=pivot_returns.index,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{val:.2f}%" for val in row] for row in pivot_returns.values],
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>"
    ))
    
    fig_heatmap.update_layout(
        title='Monthly Returns (%)',
        xaxis_title='Month',
        yaxis_title='Year',
        template='plotly_white'
    )
    
    plotly_charts['monthly_heatmap'] = fig_heatmap
    
    # 4. Rolling Metrics
    # Create subplot with 2x2 structure
    fig_rolling = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rolling Volatility (30D)', 'Rolling Sharpe (30D)', 
                       'Rolling Max Drawdown (30D)', 'Rolling Beta to Benchmark (30D)')
    )
    
    # Rolling volatility
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100  # Annualized and as percentage
    fig_rolling.add_trace(
        go.Scatter(x=rolling_vol.index, y=rolling_vol.values, 
                  mode='lines', name='Volatility',
                  line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Rolling Sharpe
    daily_rf = 0.0  # Assuming 0% risk-free rate for simplicity
    excess_returns = returns - daily_rf
    rolling_sharpe = (excess_returns.rolling(window=30).mean() / 
                     excess_returns.rolling(window=30).std()) * np.sqrt(252)
    
    fig_rolling.add_trace(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, 
                  mode='lines', name='Sharpe',
                  line=dict(color='#ff7f0e')),
        row=1, col=2
    )
    
    # Rolling Max Drawdown (using a simpler calculation)
    def rolling_max_drawdown(x):
        cumulative = (1 + x).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative / rolling_max - 1)
        return drawdown.min()
    
    rolling_dd = returns.rolling(window=30).apply(rolling_max_drawdown) * 100
    
    fig_rolling.add_trace(
        go.Scatter(x=rolling_dd.index, y=rolling_dd.values, 
                  mode='lines', name='Max Drawdown',
                  line=dict(color='#d62728')),
        row=2, col=1
    )
    
    # Rolling Beta (if benchmark exists)
    if benchmark_returns is not None:
        rolling_cov = returns.rolling(window=30).cov(benchmark_returns)
        rolling_var = benchmark_returns.rolling(window=30).var()
        rolling_beta = rolling_cov / rolling_var
        
        fig_rolling.add_trace(
            go.Scatter(x=rolling_beta.index, y=rolling_beta.values, 
                      mode='lines', name='Beta',
                      line=dict(color='#2ca02c')),
            row=2, col=2
        )
    
    fig_rolling.update_layout(
        template='plotly_white',
        showlegend=False,
        height=800,
        hovermode='x unified'
    )
    
    plotly_charts['rolling_metrics'] = fig_rolling
    
    # 5. Return Distribution
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=returns.values * 100,
        nbinsx=50,
        name='Daily Returns',
        marker_color='#1f77b4',
        opacity=0.7
    ))
    
    fig_dist.update_layout(
        title='Return Distribution',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        bargap=0.05
    )
    
    # Add a normal distribution curve for comparison
    mean = returns.mean() * 100
    std = returns.std() * 100
    x_range = np.linspace(mean - 4*std, mean + 4*std, 100)
    y_vals = stats.norm.pdf(x_range, mean, std) * len(returns) * (8*std/50)
    
    fig_dist.add_trace(go.Scatter(
        x=x_range,
        y=y_vals,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))
    
    plotly_charts['return_distribution'] = fig_dist
    
    # Return the dictionary of plotly figures
    return plotly_charts

# Generate the charts
charts = generate_plotly_charts(returns, benchmark_returns)

# Show the charts interactively
for name, fig in charts.items():
    print(f"Displaying {name} chart...")
    fig.show()

print("All charts have been generated and saved!")