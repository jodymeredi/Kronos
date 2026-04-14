"""Kronos predictor module.

Provides the KronosPredictor class for generating time-series forecasts
using a fitted KronosModel. Supports single-step and multi-step (batch)
prediction with optional confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

from .model import KronosModel


class KronosPredictor:
    """Generates price/value predictions from a trained KronosModel.

    Parameters
    ----------
    model : KronosModel
        A fitted KronosModel instance.
    confidence : float, optional
        Confidence level for prediction intervals (0 < confidence < 1).
        Defaults to 0.95.
    """

    def __init__(self, model: KronosModel, confidence: float = 0.95) -> None:
        if not hasattr(model, '_is_fitted') or not model._is_fitted:
            raise ValueError("The provided KronosModel must be fitted before prediction.")
        if not (0 < confidence < 1):
            raise ValueError("confidence must be strictly between 0 and 1.")

        self.model = model
        self.confidence = confidence

    def predict(
        self,
        steps: int = 1,
        last_known: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Predict future values for a given number of steps.

        Parameters
        ----------
        steps : int
            Number of future time steps to forecast.
        last_known : pd.Series or np.ndarray, optional
            The most recent window of observations used as the seed for
            prediction. If None, the tail of the training data is used.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['predicted', 'lower', 'upper'] indexed
            by step number (1-based).
        """
        if steps < 1:
            raise ValueError("steps must be >= 1.")

        seed = self._resolve_seed(last_known)
        predictions, lowers, uppers = [], [], []

        current_window = seed.copy()

        for _ in range(steps):
            features = self.model._build_features(current_window)
            point_pred = self.model.regressor.predict(features.reshape(1, -1))[0]

            # Denormalize prediction
            point_pred_denorm = self.model._denormalize(point_pred)

            # Estimate prediction interval via residual std
            std = self.model.residual_std_
            z = self._z_score()
            lower = point_pred_denorm - z * std
            upper = point_pred_denorm + z * std

            predictions.append(point_pred_denorm)
            lowers.append(lower)
            uppers.append(upper)

            # Slide the window forward with the raw (normalized) prediction
            current_window = np.append(current_window[1:], point_pred)

        result = pd.DataFrame(
            {
                "predicted": predictions,
                "lower": lowers,
                "upper": uppers,
            },
            index=pd.RangeIndex(start=1, stop=steps + 1, step=1),
        )
        result.index.name = "step"
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_seed(self, last_known: Optional[Union[pd.Series, np.ndarray]]) -> np.ndarray:
        """Return the normalized seed window for iterative prediction."""
        if last_known is not None:
            arr = np.asarray(last_known, dtype=float)
            if arr.ndim != 1:
                raise ValueError("last_known must be a 1-D array or Series.")
            return self.model._normalize(arr)
        # Fall back to the last window seen during training
        return self.model.last_train_window_

    def _z_score(self) -> float:
        """Return the z-score corresponding to the desired confidence level."""
        from scipy.stats import norm  # lazy import
        alpha = 1.0 - self.confidence
        return float(norm.ppf(1 - alpha / 2))
