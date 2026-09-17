from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ruddy import bootstrap_confidence_interval
from ruddy.statistics.confidence_intervals import (
    mean_confidence_interval,
    odds_ratio_confidence_interval,
    pearson_confidence_interval,
    welch_mean_difference_confidence_interval,
)


def test_bootstrap_is_deterministic() -> None:
    x = np.arange(1.0, 11.0)
    a = bootstrap_confidence_interval(x, np.mean, n_resamples=300, method="percentile", random_state=7)
    b = bootstrap_confidence_interval(x, np.mean, n_resamples=300, method="percentile", random_state=7)
    assert a == b
    assert a.status == "ok"
    assert a.confidence_low < a.estimate < a.confidence_high


def test_bootstrap_paired_requires_equal_lengths() -> None:
    with pytest.raises(ValueError, match="identical lengths"):
        bootstrap_confidence_interval(
            (np.arange(5.0), np.arange(6.0)), lambda x, y: np.mean(x - y), paired=True
        )


def test_bootstrap_rejects_too_few_resamples() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        bootstrap_confidence_interval(np.arange(10.0), np.mean, n_resamples=99)


def test_mean_interval_matches_t_formula() -> None:
    x = np.array([1.0, 2.0, 4.0, 7.0, 8.0])
    mean, se, low, high = mean_confidence_interval(x, 0.95)
    critical = stats.t.ppf(0.975, len(x) - 1)
    assert mean == pytest.approx(np.mean(x))
    assert se == pytest.approx(stats.sem(x))
    assert low == pytest.approx(mean - critical * se)
    assert high == pytest.approx(mean + critical * se)


def test_pearson_interval_is_fisher_z() -> None:
    low, high = pearson_confidence_interval(0.5, 30, 0.95)
    z = np.arctanh(0.5)
    delta = stats.norm.ppf(0.975) / np.sqrt(27)
    assert low == pytest.approx(np.tanh(z - delta))
    assert high == pytest.approx(np.tanh(z + delta))


def test_welch_mean_difference_interval() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    estimate, se, df, low, high = welch_mean_difference_confidence_interval(x, y)
    assert estimate == pytest.approx(np.mean(x) - np.mean(y))
    assert se > 0 and df > 0
    assert low < estimate < high


def test_odds_ratio_interval_without_zero_correction() -> None:
    table = np.array([[12, 5], [4, 16]])
    estimate, low, high = odds_ratio_confidence_interval(table)
    assert estimate == pytest.approx((12 * 16) / (5 * 4))
    assert low < estimate < high
    assert all(np.isnan(v) for v in odds_ratio_confidence_interval(np.array([[0, 5], [4, 16]])))
