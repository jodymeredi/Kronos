"""Utility functions for the Kronos time series prediction library."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


def validate_time_series(data: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Validate and convert input time series data to a numpy array.

    Args:
        data: Input time series data as list, pandas Series, or numpy array.

    Returns:
        A 1D numpy array of float64 values.

    Raises:
        TypeError: If data is not a supported type.
        ValueError: If data is empty or contains non-finite values.
    """
    if isinstance(data, pd.Series):
        arr = data.values.astype(np.float64)
    elif isinstance(data, (list, tuple)):
        arr = np.array(data, dtype=np.float64)
    elif isinstance(data, np.ndarray):
        arr = data.astype(np.float64)
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected list, pd.Series, or np.ndarray."
        )

    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}.")

    if len(arr) == 0:
        raise ValueError("Time series data must not be empty.")

    if not np.all(np.isfinite(arr)):
        raise ValueError("Time series data contains NaN or infinite values.")

    return arr


def compute_returns(prices: np.ndarray, log: bool = False) -> np.ndarray:
    """Compute simple or log returns from a price series.

    Args:
        prices: 1D array of price values.
        log: If True, compute log returns; otherwise compute simple returns.

    Returns:
        Array of returns (length = len(prices) - 1).
    """
    if log:
        return np.diff(np.log(prices))
    return np.diff(prices) / prices[:-1]


def rolling_window(arr: np.ndarray, window: int) -> np.ndarray:
    """Create a 2D array of rolling windows from a 1D array.

    Args:
        arr: 1D input array.
        window: Size of each rolling window.

    Returns:
        2D array of shape (len(arr) - window + 1, window).

    Raises:
        ValueError: If window size exceeds array length.
    """
    if window > len(arr):
        raise ValueError(
            f"Window size ({window}) exceeds array length ({len(arr)})."
        )
    shape = (arr.shape[0] - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def split_train_test(
    data: np.ndarray,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a time series into training and test sets.

    Args:
        data: 1D array of time series values.
        test_ratio: Fraction of data to use as the test set (default 0.2).

    Returns:
        A tuple (train, test) of numpy arrays.

    Raises:
        ValueError: If test_ratio is not in (0, 1).
    """
    if not (0 < test_ratio < 1):
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}.")

    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]


def mean_absolute_percentage_error(
    actual: np.ndarray, predicted: np.ndarray
) -> float:
    """Compute the Mean Absolute Percentage Error (MAPE).

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.

    Returns:
        MAPE as a float (e.g., 0.05 means 5%).

    Raises:
        ValueError: If arrays have different lengths or actual contains zeros.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}."
        )

    if np.any(actual == 0):
        raise ValueError("actual array contains zero values; MAPE is undefined.")

    return float(np.mean(np.abs((actual - predicted) / actual)))
