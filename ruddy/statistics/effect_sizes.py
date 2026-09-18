"""Reusable effect-size calculations for Ruddy inferential analyses."""

from __future__ import annotations

import math

import numpy as np

from ruddy.core.frames import finite_or_none


def hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float | None:
    """Bias-corrected standardized mean difference, A minus B."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = a.size, b.size
    if n_a < 2 or n_b < 2:
        return None
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    df = n_a + n_b - 2
    if df <= 0:
        return None
    pooled_variance = ((n_a - 1) * var_a + (n_b - 1) * var_b) / df
    if not math.isfinite(pooled_variance) or pooled_variance <= 0.0:
        return None
    d = (float(np.mean(a)) - float(np.mean(b))) / math.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return finite_or_none(correction * d)


def cliffs_delta_from_u(u_statistic: float, n_a: int, n_b: int) -> float | None:
    """Cliff's delta from Mann-Whitney U for group A versus group B."""
    if n_a <= 0 or n_b <= 0:
        return None
    value = 2.0 * float(u_statistic) / float(n_a * n_b) - 1.0
    if not math.isfinite(value):
        return None
    return float(max(-1.0, min(1.0, value)))


def eta_squared(groups: tuple[np.ndarray, ...]) -> float | None:
    """Descriptive eta squared from between/total sums of squares."""
    if not groups or any(np.asarray(group).size == 0 for group in groups):
        return None
    arrays = tuple(np.asarray(group, dtype=np.float64) for group in groups)
    pooled = np.concatenate(arrays)
    if pooled.size == 0:
        return None
    grand = float(np.mean(pooled))
    ss_total = float(np.sum((pooled - grand) ** 2))
    if not math.isfinite(ss_total) or ss_total <= 0.0:
        return None
    ss_between = float(sum(len(group) * (float(np.mean(group)) - grand) ** 2 for group in arrays))
    return finite_or_none(ss_between / ss_total)


def epsilon_squared(kruskal_h: float, groups: tuple[np.ndarray, ...]) -> float | None:
    """Kruskal-Wallis epsilon squared, bounded to [0, 1]."""
    k = len(groups)
    n = sum(np.asarray(group).size for group in groups)
    denominator = n - k
    if k < 2 or denominator <= 0:
        return None
    value = (float(kruskal_h) - k + 1.0) / denominator
    if not math.isfinite(value):
        return None
    return float(min(1.0, max(0.0, value)))


def bias_corrected_cramers_v(chi2: float, n: int, r: int, k: int) -> float | None:
    """Bias-corrected Cramer's V for an r x k contingency table."""
    if n <= 1 or r < 2 or k < 2:
        return None
    phi2 = float(chi2) / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denominator = min(k_corr - 1.0, r_corr - 1.0)
    if denominator <= 0.0:
        return None
    value = math.sqrt(phi2_corr / denominator)
    return float(min(1.0, max(0.0, value)))
