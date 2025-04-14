#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Quantreturns: Portfolio analytics for quants
# https://github.com/ranaroussi/quantreturns
#
# Copyright 2019-2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Portfolio visualization tools.

This module provides functions for visualizing portfolio performance metrics,
including returns, rolling statistics, drawdowns, and distributions. These
visualization tools help quantitative analysts evaluate investment strategies
compared to benchmarks and understand risk/return characteristics.
"""

from typing import Dict, List, Optional, Tuple, Union, Any


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from quantstatsv2 import stats


# Set default font
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Verdana', 'sans-serif']
except Exception:
    pass


# Define color palettes
FLATUI_COLORS = [
    "#FEDD78", "#348DC1", "#BA516B", "#4FA487", "#9B59B6",
    "#613F66", "#84B082", "#DC136C", "#559CAD", "#4A5899",
]

GRAYSCALE_COLORS = [
    "#000000", "#222222", "#555555", "#888888", "#AAAAAA",
    "#CCCCCC", "#EEEEEE", "#333333", "#666666", "#999999",
]

# Set default Seaborn style
sns.set(
    font_scale=1.1,
    rc={
        "figure.figsize": (10, 6),
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "text.color": "#333333",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
    },
)


def _get_colors(grayscale: bool) -> Tuple[List[str], str, float]:
    """Get appropriate colors, line style, and alpha based on grayscale setting.
    
    Args:
        grayscale: Whether to use grayscale colors
        
    Returns:
        Tuple of (colors, line style, alpha)
    """
    colors = FLATUI_COLORS
    ls = "-"
    alpha = 0.8
    if grayscale:
        colors = GRAYSCALE_COLORS
        alpha = 0.5
    return colors, ls, alpha


def format_pct_axis(x: float, _) -> str:
    """Format axis values as percentages with appropriate scaling.
    
    Args:
        x: Value to format
        _: Unused position argument
        
    Returns:
        Formatted percentage string
    """
    x *= 100  # Convert to percentage
    if x >= 1e12:
        res = f"{x * 1e-12:.1f}T%"
        return res.replace(".0T%", "T%")
    if x >= 1e9:
        res = f"{x * 1e-9:.1f}B%"
        return res.replace(".0B%", "B%")
    if x >= 1e6:
        res = f"{x * 1e-6:.1f}M%"
        return res.replace(".0M%", "M%")
    if x >= 1e3:
        res = f"{x * 1e-3:.1f}K%"
        return res.replace(".0K%", "K%")
    res = f"{x:.0f}%"
    return res.replace(".0%", "%")


def format_cur_axis(x: float, _) -> str:
    """Format axis values as currency with appropriate scaling.
    
    Args:
        x: Value to format
        _: Unused position argument
        
    Returns:
        Formatted currency string
    """
    if x >= 1e12:
        res = f"${x * 1e-12:.1f}T"
        return res.replace(".0T", "T")
    if x >= 1e9:
        res = f"${x * 1e-9:.1f}B"
        return res.replace(".0B", "B")
    if x >= 1e6:
        res = f"${x * 1e-6:.1f}M"
        return res.replace(".0M", "M")
    if x >= 1e3:
        res = f"${x * 1e-3:.0f}K"
        return res.replace(".0K", "K")
    res = f"${x:.0f}"
    return res.replace(".0", "")


def _setup_figure(
    figsize: Tuple[int, int] = (10, 6),
    title: str = "",
    subtitle: Optional[bool] = True,
    returns_ts = None,  # Type can be pd.Series, pd.DataFrame or None
    fontname: str = "Arial",
    ylabel: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Set up a figure with common styling.
    
    Args:
        figsize: Figure size as (width, height)
        title: Plot title
        subtitle: Whether to show subtitle with date range
        returns_ts: Return time series for date range
        fontname: Font name to use
        ylabel: Y-axis label
        
    Returns:
        Tuple of (figure, axis)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove spines
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)
    
    # Set figure colors
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    
    # Set title
    if title:
        fig.suptitle(
            title, 
            y=0.94, 
            fontweight="bold", 
            fontname=fontname, 
            fontsize=14, 
            color="black"
        )
    
    # Set subtitle with date range if requested and returns series available
    if subtitle and returns_ts is not None:
        try:
            start_date = returns_ts.index.date[0].strftime("%e %b '%y")
            end_date = returns_ts.index.date[-1].strftime("%e %b '%y")
            ax.set_title(
                f"{start_date} - {end_date}\n",
                fontsize=12,
                color="gray",
            )
        except (AttributeError, IndexError):
            pass
    
    # Set y-axis label if provided
    if ylabel:
        ax.set_ylabel(
            ylabel, 
            fontname=fontname, 
            fontweight="bold", 
            fontsize=12, 
            color="black"
        )
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    # Set x-axis label to empty
    ax.set_xlabel("")
    
    # Auto-format dates
    fig.autofmt_xdate()
    
    return fig, ax


def _finalize_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    benchmark = None,  # Type can be pd.Series, pd.DataFrame or None
    returns = None,  # Type can be pd.Series, pd.DataFrame or None
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Finalize the figure with common cleanup operations.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axis
        benchmark: Benchmark returns data
        returns: Strategy returns data
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    # Hide legend if only one series and no benchmark
    if benchmark is None and returns is not None:
        if isinstance(returns, pd.Series) or len(returns.columns) == 1:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    
    # Adjust spacing
    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass
    
    # Apply tight layout
    try:
        fig.tight_layout()
    except Exception:
        pass
    
    # Save figure if requested
    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)
    
    # Show or return figure
    if show:
        plt.show(block=False)
        plt.close()
        return None
    
    plt.close()
    return fig


def plot_returns_bars(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    returns_label: str = "Strategy",
    hline: Optional[float] = None,
    hlw: Optional[float] = None,
    hlcolor: str = "red",
    hllabel: str = "",
    resample: Optional[str] = "YE",
    title: str = "Returns",
    match_volatility: bool = False,
    log_scale: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    grayscale: bool = False,
    fontname: str = "Arial",
    ylabel: bool = True,
    subtitle: bool = True,
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot returns as a bar chart.
    
    Args:
        returns: Returns series or dataframe
        benchmark: Optional benchmark returns series
        returns_label: Label for returns series
        hline: Optional horizontal line value
        hlw: Horizontal line width
        hlcolor: Horizontal line color
        hllabel: Horizontal line label
        resample: Resampling frequency (e.g., 'YE' for year-end)
        title: Plot title
        match_volatility: Whether to match volatility of returns to benchmark
        log_scale: Whether to use logarithmic scale
        figsize: Figure size as (width, height)
        grayscale: Whether to use grayscale colors
        fontname: Font name to use
        ylabel: Whether to show y-axis label
        subtitle: Whether to show subtitle with date range
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    if match_volatility and benchmark is None:
        raise ValueError("match_volatility requires passing of benchmark.")
    
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    # Prepare data
    colors, _, _ = _get_colors(grayscale)
    
    if isinstance(returns, pd.Series):
        df = pd.DataFrame(index=returns.index, data={returns.name: returns})
    elif isinstance(returns, pd.DataFrame):
        df = pd.DataFrame(
            index=returns.index, 
            data={col: returns[col] for col in returns.columns}
        )
        
    if isinstance(benchmark, pd.Series):
        df[benchmark.name] = benchmark[benchmark.index.isin(returns.index)]
        if isinstance(returns, pd.Series):
            df = df[[benchmark.name, returns.name]]
        elif isinstance(returns, pd.DataFrame):
            col_names = [benchmark.name] + list(returns.columns)
            df = df[col_names]

    df = df.dropna()
    
    # Resample data if requested
    if resample is not None:
        df = df.resample(resample).apply(stats.comp).resample(resample).last()

    # Set up figure
    fig, ax = _setup_figure(
        figsize=figsize,
        title=title,
        subtitle=subtitle,
        returns_ts=df,
        fontname=fontname,
        ylabel="Returns" if ylabel else None,
    )

    # Plotting
    if benchmark is None:
        colors = colors[1:]
    df.plot(kind="bar", ax=ax, color=colors)

    # Set x-axis labels
    try:
        ax.set_xticklabels(df.index.year)
        years = sorted(list(set(df.index.year)))
    except AttributeError:
        ax.set_xticklabels(df.index)
        years = sorted(list(set(df.index)))

    # Limit number of x-tick labels if too many
    if len(years) > 10:
        mod = int(len(years) / 10)
        plt.xticks(
            np.arange(len(years)),
            [str(year) if not i % mod else "" for i, year in enumerate(years)],
        )

    # Add horizontal lines
    if hline is not None:
        if not isinstance(hline, pd.Series):
            if grayscale:
                hlcolor = "gray"
            ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    # Set y-axis formatter and scale
    plt.yscale("symlog" if log_scale else "linear")
    ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

    # Add functions for longest drawdowns and distribution plots
    return _finalize_figure(
        fig=fig,
        ax=ax,
        benchmark=benchmark,
        returns=returns,
        savefig=savefig,
        show=show,
    )


def plot_longest_drawdowns(
    returns: pd.Series,
    periods: int = 5,
    lw: float = 1.5,
    fontname: str = "Arial",
    grayscale: bool = False,
    title: Optional[str] = None,
    log_scale: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot the longest drawdown periods.
    
    Args:
        returns: Returns series
        periods: Number of drawdown periods to plot
        lw: Line width
        fontname: Font name to use
        grayscale: Whether to use grayscale colors
        title: Plot title
        log_scale: Whether to use logarithmic scale
        figsize: Figure size as (width, height)
        ylabel: Whether to show y-axis label
        subtitle: Whether to show subtitle with date range
        compounded: Whether to show compounded returns
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    colors = ["#348dc1", "#003366", "red"]
    if grayscale:
        colors = ["#000000"] * 3

    # Calculate drawdowns
    dd = stats.to_drawdown_series(returns.fillna(0))
    dddf = stats.drawdown_details(dd)
    longest_dd = dddf.sort_values(by="days", ascending=False, kind="mergesort")[:periods]

    # Prepare figure title
    if title is None:
        title = "Worst Drawdown Periods"
    else:
        title = f"{title} - Worst Drawdown Periods"

    # Set up figure
    fig, ax = _setup_figure(
        figsize=figsize,
        title=title,
        subtitle=subtitle,
        returns_ts=returns,
        fontname=fontname,
        ylabel="Cumulative Returns" if ylabel else None,
    )

    # Plot cumulative returns
    series = stats.compsum(returns) if compounded else returns.cumsum()
    ax.plot(series, lw=lw, label="Backtest", color=colors[0])

    # Highlight drawdown periods
    highlight = "black" if grayscale else "red"
    for _, row in longest_dd.iterrows():
        ax.axvspan(
            *mdates.datestr2num([str(row["start"]), str(row["end"])]),
            color=highlight,
            alpha=0.1,
        )

    # Add zero line
    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)
    
    # Set scale and formatter
    plt.yscale("symlog" if log_scale else "linear")
    ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

    # Finalize and return
    return _finalize_figure(
        fig=fig,
        ax=ax,
        benchmark=None,
        returns=returns,
        savefig=savefig,
        show=show,
    )


def plot_distribution(
    returns: pd.Series,
    figsize: Tuple[int, int] = (10, 6),
    fontname: str = "Arial",
    grayscale: bool = False,
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    title: Optional[str] = None,
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot the distribution of returns across different time periods.
    
    Args:
        returns: Returns series
        figsize: Figure size as (width, height)
        fontname: Font name to use
        grayscale: Whether to use grayscale colors
        ylabel: Whether to show y-axis label
        subtitle: Whether to show subtitle with date range
        compounded: Whether to compound returns
        title: Plot title
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    # Set colors
    colors = FLATUI_COLORS
    if grayscale:
        colors = ["#f9f9f9", "#dddddd", "#bbbbbb", "#999999", "#808080"]

    # Prepare data
    port = pd.DataFrame(returns.fillna(0))
    port.columns = ["Daily"]

    # Select appropriate function for resampling
    apply_fnc = stats.comp if compounded else np.sum

    # Create different time period series
    port["Weekly"] = port["Daily"].resample("W-MON").apply(apply_fnc)
    port["Weekly"].ffill(inplace=True)

    port["Monthly"] = port["Daily"].resample("ME").apply(apply_fnc)
    port["Monthly"].ffill(inplace=True)

    port["Quarterly"] = port["Daily"].resample("QE").apply(apply_fnc)
    port["Quarterly"].ffill(inplace=True)

    port["Yearly"] = port["Daily"].resample("YE").apply(apply_fnc)
    port["Yearly"].ffill(inplace=True)

    # Prepare title
    if title:
        title = f"{title} - Return Quantiles"
    else:
        title = "Return Quantiles"

    # Set up figure
    fig, ax = _setup_figure(
        figsize=figsize,
        title=title,
        subtitle=subtitle,
        returns_ts=returns,
        fontname=fontname,
        ylabel="Returns" if ylabel else None,
    )

    # Create box plot
    sns.boxplot(
        data=port,
        ax=ax,
        palette={
            "Daily": colors[0],
            "Weekly": colors[1],
            "Monthly": colors[2],
            "Quarterly": colors[3],
            "Yearly": colors[4],
        },
    )

    # Set formatter
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{int(x * 100):,}%")
    )

    # Finalize and return
    return _finalize_figure(
        fig=fig,
        ax=ax,
        benchmark=None,
        returns=returns,
        savefig=savefig,
        show=show,
    )


def plot_table(
    tbl: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "",
    title_loc: str = "left",
    header: bool = True,
    colWidths: Optional[List[float]] = None,
    rowLoc: str = "right",
    colLoc: str = "right",
    colLabels: Optional[List[str]] = None,
    edges: str = "horizontal",
    orient: str = "horizontal",
    figsize: Tuple[float, float] = (5.5, 6),
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Plot a table of data.
    
    Args:
        tbl: DataFrame to plot as table
        columns: Optional column names to rename DataFrame columns
        title: Plot title
        title_loc: Title location
        header: Whether to show header
        colWidths: List of column widths
        rowLoc: Location of row labels
        colLoc: Location of column labels
        colLabels: Column labels
        edges: Edge style (horizontal, vertical, both, none)
        orient: Orientation (horizontal, vertical)
        figsize: Figure size as (width, height)
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    # Update column names if provided
    if columns is not None:
        try:
            tbl.columns = columns
        except Exception:
            pass

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, frame_on=False)

    # Add title
    if title:
        ax.set_title(
            title, fontweight="bold", fontsize=14, color="black", loc=title_loc
        )

    # Create table
    the_table = ax.table(
        cellText=tbl.values,
        colWidths=colWidths,
        rowLoc=rowLoc,
        colLoc=colLoc,
        edges=edges,
        colLabels=(tbl.columns if header else colLabels),
        loc="center",
        zorder=2,
    )

    # Customize table appearance
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1)

    # Style cells
    for (row, col), cell in the_table.get_celld().items():
        cell.set_height(0.08)
        cell.set_text_props(color="black")
        cell.set_edgecolor("#dddddd")
        if row == 0 and header:
            cell.set_edgecolor("black")
            cell.set_facecolor("black")
            cell.set_linewidth(2)
            cell.set_text_props(weight="bold", color="black")
        elif col == 0 and "vertical" in orient:
            cell.set_edgecolor("#dddddd")
            cell.set_linewidth(1)
            cell.set_text_props(weight="bold", color="black")
        elif row > 1:
            cell.set_linewidth(1)

    # Hide grid and ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust layout
    try:
        plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    # Save figure if requested
    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    # Show or return figure
    if show:
        plt.show(block=False)
        plt.close()
        return None
    
    plt.close()
    return fig


def plot_timeseries(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    title: str = "Returns",
    compound: bool = False,
    cumulative: bool = True,
    fill: bool = False,
    returns_label: str = "Strategy",
    hline: Optional[float] = None,
    hlw: Optional[float] = None,
    hlcolor: str = "red",
    hllabel: str = "",
    percent: bool = True,
    match_volatility: bool = False,
    log_scale: bool = False,
    resample: Optional[str] = None,
    lw: float = 1.5,
    figsize: Tuple[int, int] = (10, 6),
    ylabel: str = "",
    grayscale: bool = False,
    fontname: str = "Arial",
    subtitle: bool = True,
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot returns as a time series.
    
    Args:
        returns: Returns series or dataframe
        benchmark: Optional benchmark returns series
        title: Plot title
        compound: Whether to compound returns
        cumulative: Whether to show cumulative returns
        fill: Whether to fill area under curves
        returns_label: Label for returns series
        hline: Optional horizontal line value
        hlw: Horizontal line width
        hlcolor: Horizontal line color
        hllabel: Horizontal line label
        percent: Whether to format y-axis as percentages
        match_volatility: Whether to match volatility of returns to benchmark
        log_scale: Whether to use logarithmic scale
        resample: Resampling frequency
        lw: Line width
        figsize: Figure size as (width, height)
        ylabel: Y-axis label
        grayscale: Whether to use grayscale colors
        fontname: Font name to use
        subtitle: Whether to show subtitle with date range
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    colors, ls, alpha = _get_colors(grayscale)

    # Handle missing values
    returns = returns.fillna(0)
    if isinstance(benchmark, pd.Series):
        benchmark = benchmark.fillna(0)

    if match_volatility and benchmark is None:
        raise ValueError("match_volatility requires passing of benchmark.")
    
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.std()
        returns = (returns / returns.std()) * bmark_vol

    # Apply transformations based on parameters
    if compound is True:
        if cumulative:
            returns = stats.compsum(returns)
            if isinstance(benchmark, pd.Series):
                benchmark = stats.compsum(benchmark)
        else:
            returns = returns.cumsum()
            if isinstance(benchmark, pd.Series):
                benchmark = benchmark.cumsum()

    # Find the resampling code (likely in core.py):

# Replace this:
    if resample:
        returns = returns.resample(resample) if compound else returns.resample(resample).sum()
        
        if isinstance(benchmark, pd.Series):
            benchmark = benchmark.resample(resample).last()
            
    # Set up figure
    fig, ax = _setup_figure(
        figsize=figsize,
        title=title,
        subtitle=subtitle,
        returns_ts=returns,
        fontname=fontname,
        ylabel=ylabel,
    )

    # Plot benchmark if provided
    if isinstance(benchmark, pd.Series):
        ax.plot(benchmark, lw=lw, ls=ls, label=benchmark.name, color=colors[0])

    # Plot returns
    plot_alpha = 0.25 if grayscale else 1
    if isinstance(returns, pd.Series):
        ax.plot(returns, lw=lw, label=returns.name, color=colors[1], alpha=plot_alpha)
        if fill:
            ax.fill_between(returns.index, 0, returns, color=colors[1], alpha=0.25)
    elif isinstance(returns, pd.DataFrame):
        for i, col in enumerate(returns.columns):
            ax.plot(returns[col], lw=lw, label=col, alpha=plot_alpha, color=colors[i + 1])
            if fill:
                ax.fill_between(returns[col].index, 0, returns[col], color=colors[i + 1], alpha=0.25)

    # Add horizontal lines
    if hline is not None:
        if not isinstance(hline, pd.Series):
            if grayscale:
                hlcolor = "black"
            ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    ax.axhline(0, ls="-", lw=1, color="gray", zorder=1)
    ax.axhline(0, ls="--", lw=1, color="white" if grayscale else "black", zorder=2)

    # Set y-axis formatter and scale
    plt.yscale("symlog" if log_scale else "linear")
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

    # Finalize and return
    return _finalize_figure(
        fig=fig,
        ax=ax,
        benchmark=benchmark,
        returns=returns,
        savefig=savefig,
        show=show,
    )


def plot_histogram(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    resample: str = "ME",
    bins: int = 20,
    fontname: str = "Arial",
    grayscale: bool = False,
    title: str = "Returns",
    kde: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot a histogram of returns.
    
    Args:
        returns: Returns series or dataframe
        benchmark: Optional benchmark returns series
        resample: Resampling frequency (e.g., 'ME' for month-end)
        bins: Number of histogram bins
        fontname: Font name to use
        grayscale: Whether to use grayscale colors
        title: Plot title
        kde: Whether to show kernel density estimation
        figsize: Figure size as (width, height)
        ylabel: Whether to show y-axis label
        subtitle: Whether to show subtitle with date range
        compounded: Whether to compound returns
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    colors, _, _ = _get_colors(grayscale)

    # Apply function based on compounded parameter
    apply_fnc = stats.comp if compounded else np.sum
    
    # Prepare benchmark data if provided
    if benchmark is not None:
        benchmark = (
            benchmark.fillna(0)
            .resample(resample)
            .apply(apply_fnc)
            .resample(resample)
            .last()
        )

    # Prepare returns data
    returns = (
        returns.fillna(0)
        .resample(resample)
        .apply(apply_fnc)
        .resample(resample)
        .last()
    )

    # Adjust figure size slightly
    figsize = (0.995 * figsize[0], figsize[1])
    
    # Set up figure
    fig, ax = _setup_figure(
        figsize=figsize,
        title=title,
        subtitle=subtitle,
        returns_ts=returns,
        fontname=fontname,
        ylabel="Occurrences" if ylabel else None,
    )

    # Convert single column DataFrame to Series for easier handling
    if isinstance(returns, pd.DataFrame) and len(returns.columns) == 1:
        returns = returns[returns.columns[0]]

    # Set up color palette
    pallete = colors[1:2] if benchmark is None else colors[:2]
    alpha = 0.7
    
    if isinstance(returns, pd.DataFrame):
        pallete = (
            colors[1 : len(returns.columns) + 1]
            if benchmark is None
            else colors[: len(returns.columns) + 1]
        )
        if len(returns.columns) > 1:
            alpha = 0.5

    # Create plot based on input data types
    if benchmark is not None:
        if isinstance(returns, pd.Series):
            combined_returns = (
                benchmark.to_frame()
                .join(returns.to_frame())
                .stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
        elif isinstance(returns, pd.DataFrame):
            combined_returns = (
                benchmark.to_frame()
                .join(returns)
                .stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
        sns.histplot(
            data=combined_returns,
            x="Returns",
            bins=bins,
            alpha=alpha,
            kde=kde,
            stat="density",
            hue="",
            palette=pallete,
            ax=ax,
        )
    else:
        if isinstance(returns, pd.Series):
            combined_returns = returns.copy()
            if kde:
                sns.kdeplot(data=combined_returns, color="black", ax=ax)
            sns.histplot(
                data=combined_returns,
                bins=bins,
                alpha=alpha,
                kde=False,
                stat="density",
                color=colors[1],
                ax=ax,
            )
        elif isinstance(returns, pd.DataFrame):
            combined_returns = (
                returns.stack()
                .reset_index()
                .rename(columns={"level_1": "", 0: "Returns"})
            )
            sns.histplot(
                data=combined_returns,
                x="Returns",
                bins=bins,
                alpha=alpha,
                kde=kde,
                stat="density",
                hue="",
                palette=pallete,
                ax=ax,
            )

    # Add average line for single series
    if isinstance(combined_returns, pd.Series) or isinstance(combined_returns, pd.DataFrame) and len(combined_returns.columns) == 1:
        ax.axvline(
            combined_returns.mean(),
            ls="--",
            lw=1.5,
            zorder=2,
            label="Average",
            color="red",
        )

    # Set axis formatters
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{int(x * 100):,}%")
    )

    # Finalize and return
    return _finalize_figure(
        fig=fig,
        ax=ax,
        benchmark=benchmark,
        returns=returns,
        savefig=savefig,
        show=show,
    )


def plot_rolling_stats(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    title: str = "",
    returns_label: Union[str, List[str]] = "Strategy",
    hline: Optional[float] = None,
    hlw: Optional[float] = None,
    hlcolor: str = "red",
    hllabel: str = "",
    lw: float = 1.5,
    figsize: Tuple[int, int] = (10, 6),
    ylabel: str = "",
    grayscale: bool = False,
    fontname: str = "Arial",
    subtitle: bool = True,
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot rolling statistics.
    
    Args:
        returns: Returns series or dataframe
        benchmark: Optional benchmark returns series
        title: Plot title
        returns_label: Label for returns series or list of labels
        hline: Optional horizontal line value
        hlw: Horizontal line width
        hlcolor: Horizontal line color
        hllabel: Horizontal line label
        lw: Line width
        figsize: Figure size as (width, height)
        ylabel: Y-axis label
        grayscale: Whether to use grayscale colors
        fontname: Font name to use
        subtitle: Whether to show subtitle with date range
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    colors, _, _ = _get_colors(grayscale)

    # Set up figure
    fig, ax = _setup_figure(
        figsize=figsize,
        title=title,
        subtitle=subtitle,
        returns_ts=returns,
        fontname=fontname,
        ylabel=ylabel,
    )

    # Handle different input types and prepare data
    if isinstance(returns, pd.DataFrame):
        if isinstance(returns_label, str):
            returns_label = list(returns.columns)

    # Create DataFrame for plotting
    if isinstance(returns, pd.Series):
        df = pd.DataFrame(index=returns.index, data={returns_label: returns})
    elif isinstance(returns, pd.DataFrame):
        df = pd.DataFrame(
            index=returns.index, data={col: returns[col] for col in returns.columns}
        )
    
    # Add benchmark if provided
    if isinstance(benchmark, pd.Series):
        df["Benchmark"] = benchmark[benchmark.index.isin(returns.index)]
        if isinstance(returns, pd.Series):
            df = df[["Benchmark", returns_label]].dropna()
            ax.plot(
                df[returns_label].dropna(), lw=lw, label=returns.name, color=colors[1]
            )
        elif isinstance(returns, pd.DataFrame):
            col_names = ["Benchmark"] + (returns_label if isinstance(returns_label, list) else [returns_label])
            df = df[col_names].dropna()
            for i, col in enumerate(returns_label):
                ax.plot(df[col], lw=lw, label=col, color=colors[i + 1])
        ax.plot(
            df["Benchmark"], lw=lw, label=benchmark.name, color=colors[0], alpha=0.8
        )
    else:
        if isinstance(returns, pd.Series):
            df = df[[returns_label]].dropna()
            ax.plot(
                df[returns_label].dropna(), lw=lw, label=returns.name, color=colors[1]
            )
        elif isinstance(returns, pd.DataFrame):
            df = df[returns_label].dropna()
            for i, col in enumerate(returns_label):
                ax.plot(df[col], lw=lw, label=col, color=colors[i + 1])

    # Add horizontal lines
    if hline is not None:
        if not isinstance(hline, pd.Series):
            if grayscale:
                hlcolor = "black"
            ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    # Set axis formatters
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Finalize and return
    return _finalize_figure(
        fig=fig,
        ax=ax,
        benchmark=benchmark,
        returns=returns,
        savefig=savefig,
        show=show,
    )


def plot_rolling_beta(
    returns: Union[pd.Series, pd.DataFrame],
    benchmark: pd.Series,
    window1: int = 126,
    window1_label: str = "",
    window2: Optional[int] = None,
    window2_label: str = "",
    title: str = "",
    hlcolor: str = "red",
    figsize: Tuple[int, int] = (10, 6),
    grayscale: bool = False,
    fontname: str = "Arial",
    lw: float = 1.5,
    ylabel: bool = True,
    subtitle: bool = True,
    savefig: Optional[Union[str, Dict[str, Any]]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot rolling beta.
    
    Args:
        returns: Returns series or dataframe
        benchmark: Benchmark returns series
        window1: First rolling window length
        window1_label: Label for first window
        window2: Optional second rolling window length
        window2_label: Label for second window
        title: Plot title
        hlcolor: Horizontal line color
        figsize: Figure size as (width, height)
        grayscale: Whether to use grayscale colors
        fontname: Font name to use
        lw: Line width
        ylabel: Whether to show y-axis label
        subtitle: Whether to show subtitle with date range
        savefig: Path to save figure or dict of options
        show: Whether to show the figure
        
    Returns:
        Figure object if show=False, otherwise None
    """
    colors, _, _ = _get_colors(grayscale)

    # Set up figure
    fig, ax = _setup_figure(
        figsize=figsize,
        title=title,
        subtitle=subtitle,
        returns_ts=returns,
        fontname=fontname,
        ylabel="Beta" if ylabel else None,
    )

    # Plot first window beta
    i = 1
    if isinstance(returns, pd.Series):
        beta = stats.rolling_greeks(returns, benchmark, window1)["beta"].fillna(0)
        ax.plot(beta, lw=lw, label=window1_label, color=colors[1])
    elif isinstance(returns, pd.DataFrame):
        beta = {
            col: stats.rolling_greeks(returns[col], benchmark, window1)["beta"].fillna(0)
            for col in returns.columns
        }
        for name, b in beta.items():
            ax.plot(b, lw=lw, label=f"{name} ({window1_label})", color=colors[i])
            i += 1

    # Plot second window beta if provided
    i = 1
    if window2:
        reduced_lw = lw - 0.5
        if isinstance(returns, pd.Series):
            beta2 = stats.rolling_greeks(returns, benchmark, window2)["beta"]
            ax.plot(
                beta2,
                lw=reduced_lw,
                label=window2_label,
                color="gray",
                alpha=0.8,
            )
        elif isinstance(returns, pd.DataFrame):
            betas_w2 = {
                col: stats.rolling_greeks(returns[col], benchmark, window2)["beta"]
                for col in returns.columns
            }
            for name, beta_w2 in betas_w2.items():
                ax.plot(
                    beta_w2,
                    lw=reduced_lw,
                    ls="--",
                    label=f"{name} ({window2_label})",
                    alpha=0.5,
                    color=colors[i],
                )
                i += 1

    # Set appropriate y-axis ticks
    beta_min = (
        beta.min()
        if isinstance(returns, pd.Series)
        else min([b.min() for b in beta.values()])
    )
    beta_max = (
        beta.max()
        if isinstance(returns, pd.Series)
        else max([b.max() for b in beta.values()])
    )
    mmin = min([-100, int(beta_min * 100)])
    mmax = max([100, int(beta_max * 100)])
    step = 50 if (mmax - mmin) >= 200 else 100
    ax.set_yticks([x / 100 for x in list(range(mmin, mmax, step))])

    # Add mean line for series
    if isinstance(returns, pd.Series):
        hlcolor = "black" if grayscale else hlcolor
        ax.axhline(beta.mean(), ls="--", lw=1.5, color=hlcolor, zorder=2)

    # Add zero line
    ax.axhline(0, ls="--", lw=1, color="#000000", zorder=2)

    # Set x-axis formatter for precise dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Finalize and return
    return _finalize_figure(
        fig=fig,
        ax=ax,
        benchmark=benchmark,
        returns=returns,
        savefig=savefig,
        show=show,
    )