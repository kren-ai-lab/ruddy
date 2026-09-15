"""Analytic confidence-interval primitives used across Ruddy."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def mean_confidence_interval(
    values: np.ndarray, confidence_level: float = 0.95
) -> tuple[float, float, float, float]:
    """Return mean, standard error, lower CI, and upper CI using Student's t."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        nan = float("nan")
        return nan, nan, nan, nan
    mean = float(np.mean(x))
    se = float(stats.sem(x))
    critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, x.size - 1))
    return mean, se, mean - critical * se, mean + critical * se


def pearson_confidence_interval(r: float, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Fisher-z confidence interval for Pearson's correlation."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    critical = float(stats.norm.ppf((1.0 + confidence_level) / 2.0))
    return float(np.tanh(z - critical * se)), float(np.tanh(z + critical * se))


def welch_mean_difference_confidence_interval(
    x: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float, float]:
    """Welch interval for mean(x) - mean(y)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        nan = float("nan")
        return nan, nan, nan, nan, nan
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    estimate = float(np.mean(a) - np.mean(b))
    se2 = va / a.size + vb / b.size
    if se2 <= 0.0:
        return estimate, 0.0, float("inf"), estimate, estimate
    se = math.sqrt(se2)
    numerator = se2**2
    denominator = (va / a.size) ** 2 / (a.size - 1) + (vb / b.size) ** 2 / (b.size - 1)
    df = numerator / denominator if denominator > 0.0 else float("inf")
    critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df))
    return estimate, se, float(df), estimate - critical * se, estimate + critical * se


def odds_ratio_confidence_interval(
    table: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Log-Wald interval for a 2x2 odds ratio without hidden zero-cell correction."""
    counts = np.asarray(table, dtype=float)
    if counts.shape != (2, 2) or np.any(counts <= 0.0):
        return float("nan"), float("nan"), float("nan")
    a, b, c, d = counts.ravel()
    odds_ratio = float((a * d) / (b * c))
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    critical = float(stats.norm.ppf((1.0 + confidence_level) / 2.0))
    log_or = math.log(odds_ratio)
    return odds_ratio, math.exp(log_or - critical * se), math.exp(log_or + critical * se)
