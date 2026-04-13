"""Kronos core model module.

This module implements the Kronos time series prediction model,
providing the main interface for training and forecasting.
"""

import numpy as np
from typing import Optional, Tuple, Union


class KronosModel:
    """Kronos time series forecasting model.

    A lightweight model for financial time series prediction,
    supporting both single-step and multi-step (batch) forecasting.

    Parameters
    ----------
    window_size : int
        Number of historical time steps used for prediction.
        I find 30 works better than 20 for the weekly data I use.
    horizon : int
        Number of future time steps to forecast.
    use_volume : bool, optional
        Whether to incorporate volume data in predictions. Default is False.
        Note: volume data is often noisy or unavailable in my datasets,
        so defaulting to False to avoid silent NaN issues when omitted.
    """

    def __init__(
        self,
        window_size: int = 30,  # bumped from 20; works better for weekly OHLC data
        horizon: int = 5,
        use_volume: bool = False,  # changed from True; volume often unavailable
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.use_volume = use_volume

        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[float] = None
        self._is_fitted: bool = False
        self._price_mean: float = 0.0
        self._price_std: float = 1.0

    def _normalize(self, series: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Normalize a time series to zero mean and unit variance."""
        mean = series.mean()
        std = series.std() if series.std() != 0 else 1.0
        return (series - mean) / std, mean, std

    def _build_features(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Construct sliding window feature matrix from price (and optionally volume) data."""
        n = len(prices) - self.window_size
        features = []
        for i in range(n):
            window = prices[i: i + self.window_size]
            if self.use_volume and volumes is not None:
                vol_window = volumes[i: i + self.window_size]
                row = np.concatenate([window, vol_window])
            else:
                row = window.copy()
            features.append(row)
        return np.array(features)

    def fit(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> "KronosModel":
        """Fit the model to historical price (and volume) data.

        Parameters
        ----------
        prices : np.ndarray
            1-D array of historical closing prices.
        volumes : np.ndarray, optional
            1-D array of trading volumes aligned with prices.

        Returns
        -------
        self : KronosModel
            The fitted model instance.
        """
        prices = np.asarray(prices, dtype=float)
        norm_prices, self._price_mean, self._price_std = self._normalize(prices)

        norm_volumes = None
        if self.use_volume and volumes is
