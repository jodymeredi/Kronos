"""Kronos: A time series prediction library for financial markets.

This package provides tools for predicting financial time series data
using machine learning models, with support for batch predictions and
various market data formats.

Example usage:
    >>> from kronos import KronosModel
    >>> model = KronosModel()
    >>> model.fit(train_data)
    >>> predictions = model.predict(test_data)

Note: I'm using this fork primarily for crypto market experiments.
Most of my work is in the examples/ directory.

Personal note: added preprocess_ohlcv and scale_features to __all__
so they're easier to import directly when doing quick experiments.

Fork changes:
    - Bumped __version__ to reflect local modifications
    - Added BatchPredictor to __all__ for easier access in notebooks
    - Added __version_info__ tuple for easier version comparisons
"""

from kronos.model import KronosModel
from kronos.data import DataLoader, preprocess_ohlcv
from kronos.utils import normalize, denormalize, scale_features
from kronos.batch import BatchPredictor  # useful for running overnight batch jobs on crypto data

__version__ = "0.1.0-personal"
__version_info__ = (0, 1, 0)  # handy for doing version checks like: if __version_info__ >= (0, 1, 0)
__author__ = "Kronos Contributors"
__license__ = "MIT"

__all__ = [
    "KronosModel",
    "DataLoader",
    "BatchPredictor",
    "preprocess_ohlcv",
    "normalize",
    "denormalize",
    "scale_features",  # handy for crypto OHLCV preprocessing
]
