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
    horizon : int
        Number of future time steps to forecast.
    use_volume : bool, optional
        Whether to incorporate volume data in predictions. Default is True.
    """

    def __init__(
        self,
        window_size: int = 20,
        horizon: int = 5,
        use_volume: bool = True,
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
        if self.use_volume and volumes is not None:
            volumes = np.asarray(volumes, dtype=float)
            norm_volumes, _, _ = self._normalize(volumes)

        X = self._build_features(norm_prices, norm_volumes)
        # Target: next price after each window
        y = norm_prices[self.window_size:]

        # Closed-form least squares: w = (X^T X)^{-1} X^T y
        XtX = X.T @ X
        reg = 1e-4 * np.eye(XtX.shape[0])  # L2 regularisation for stability
        self._weights = np.linalg.solve(XtX + reg, X.T @ y)
        self._bias = 0.0
        self._is_fitted = True
        return self

    def predict(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate multi-step price forecasts.

        Parameters
        ----------
        prices : np.ndarray
            Recent price history; must contain at least `window_size` values.
        volumes : np.ndarray, optional
            Recent volume history aligned with prices.

        Returns
        -------
        forecast : np.ndarray
            Array of length `horizon` with predicted prices.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        prices = np.asarray(prices, dtype=float)
        if len(prices) < self.window_size:
            raise ValueError(
                f"prices must have at least {self.window_size} elements, got {len(prices)}."
            )

        norm_prices = (prices - self._price_mean) / self._price_std
        norm_volumes = None
        if self.use_volume and volumes is not None:
            volumes = np.asarray(volumes, dtype=float)
            norm_volumes, _, _ = self._normalize(volumes)

        forecast_norm = []
        price_buf = norm_prices[-self.window_size:].tolist()
        vol_buf = (
            norm_volumes[-self.window_size:].tolist()
            if norm_volumes is not None
            else None
        )

        for _ in range(self.horizon):
            window = np.array(price_buf[-self.window_size:])
            if self.use_volume and vol_buf is not None:
                vol_window = np.array(vol_buf[-self.window_size:])
                feat = np.concatenate([window, vol_window])
            else:
                feat = window
            next_norm = float(feat @ self._weights) + self._bias
            forecast_norm.append(next_norm)
            price_buf.append(next_norm)
            if vol_buf is not None:
                # Carry forward last volume as a naive estimate
                vol_buf.append(vol_buf[-1])

        forecast = np.array(forecast_norm) * self._price_std + self._price_mean
        return forecast

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"KronosModel(window_size={self.window_size}, "
            f"horizon={self.horizon}, use_volume={self.use_volume}, "
            f"status={status})"
        )
