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
"""

from kronos.model import KronosModel
from kronos.data import DataLoader, preprocess_ohlcv
from kronos.utils import normalize, denormalize

__version__ = "0.1.0"
__author__ = "Kronos Contributors"
__license__ = "MIT"

__all__ = [
    "KronosModel",
    "DataLoader",
    "preprocess_ohlcv",
    "normalize",
    "denormalize",
]
