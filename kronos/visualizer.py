"""Visualization utilities for Kronos predictions and model diagnostics."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Sequence


def plot_forecast(
    historical: np.ndarray,
    predicted: np.ndarray,
    actual: Optional[np.ndarray] = None,
    timestamps: Optional[Sequence] = None,
    title: str = "Kronos Forecast",
    figsize: tuple = (12, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot historical prices alongside the model's forecast.

    Parameters
    ----------
    historical : np.ndarray
        Array of historical price values used as model input.
    predicted : np.ndarray
        Array of forecasted price values.
    actual : np.ndarray, optional
        Ground-truth future values for comparison.
    timestamps : sequence, optional
        Datetime-like labels for the combined series axis.
    title : str
        Plot title.
    figsize : tuple
        Figure size when a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on; a new figure is created if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    n_hist = len(historical)
    n_pred = len(predicted)

    if timestamps is not None:
        hist_x = list(timestamps[:n_hist])
        pred_x = list(timestamps[n_hist: n_hist + n_pred])
    else:
        hist_x = list(range(n_hist))
        pred_x = list(range(n_hist, n_hist + n_pred))

    ax.plot(hist_x, historical, label="Historical", color="steelblue", linewidth=1.5)
    ax.plot(
        [hist_x[-1]] + pred_x,
        np.concatenate([[historical[-1]], predicted]),
        label="Forecast",
        color="darkorange",
        linewidth=1.5,
        linestyle="--",
    )

    if actual is not None:
        ax.plot(
            pred_x,
            actual[:n_pred],
            label="Actual",
            color="seagreen",
            linewidth=1.5,
            alpha=0.8,
        )

    if timestamps is not None:
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Plot residuals (actual - predicted) to diagnose forecast errors.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth values.
    predicted : np.ndarray
        Model forecast values aligned with *actual*.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = np.asarray(actual) - np.asarray(predicted)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(residuals, color="tomato", linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Residuals over time")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
