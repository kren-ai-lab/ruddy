"""Robust statistical primitives used by Ruddy diagnostics."""

from __future__ import annotations

import numpy as np

MODIFIED_Z_CONSISTENCY: float = 0.6744897501960817


def median_absolute_deviation(values: np.ndarray) -> float:
    """Return the unscaled median absolute deviation of finite numeric values."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        msg = "values must be one-dimensional."
        raise ValueError(msg)
    if array.size == 0:
        msg = "values cannot be empty."
        raise ValueError(msg)
    if not np.isfinite(array).all():
        msg = "values must contain only finite observations."
        raise ValueError(msg)
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def modified_z_scores(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return modified Z-scores together with their median and unscaled MAD."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        msg = "values must be one-dimensional."
        raise ValueError(msg)
    if array.size == 0:
        msg = "values cannot be empty."
        raise ValueError(msg)
    if not np.isfinite(array).all():
        msg = "values must contain only finite observations."
        raise ValueError(msg)
    median = float(np.median(array))
    mad = median_absolute_deviation(array)
    if mad <= 0.0:
        msg = "modified Z-scores are undefined when MAD is zero."
        raise ValueError(msg)
    scores = MODIFIED_Z_CONSISTENCY * (array - median) / mad
    return scores.astype(float, copy=False), median, mad


def tukey_fences(
    values: np.ndarray,
    *,
    multiplier: float = 1.5,
) -> tuple[float, float, float, float]:
    """Return Q1, Q3 and Tukey lower/upper fences for finite observations."""
    if multiplier <= 0:
        msg = "multiplier must be greater than zero."
        raise ValueError(msg)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        msg = "values must be one-dimensional."
        raise ValueError(msg)
    if array.size == 0:
        msg = "values cannot be empty."
        raise ValueError(msg)
    if not np.isfinite(array).all():
        msg = "values must contain only finite observations."
        raise ValueError(msg)
    q25, q75 = np.quantile(array, [0.25, 0.75])
    iqr = float(q75 - q25)
    lower = float(q25 - multiplier * iqr)
    upper = float(q75 + multiplier * iqr)
    return float(q25), float(q75), lower, upper
