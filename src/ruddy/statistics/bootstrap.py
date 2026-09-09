"""Deterministic bootstrap confidence intervals for reusable Ruddy statistics."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Callable, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Compact bootstrap result without retaining the full resample distribution."""

    estimate: float
    confidence_low: float
    confidence_high: float
    standard_error: float
    confidence_level: float
    n_resamples: int
    method: str
    random_state: int
    status: str = "ok"
    reason: str | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "estimate": self.estimate,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "standard_error": self.standard_error,
            "confidence_level": self.confidence_level,
            "n_resamples": self.n_resamples,
            "method": self.method,
            "random_state": self.random_state,
            "status": self.status,
            "reason": self.reason,
        }


def _as_1d_finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Bootstrap inputs must be one-dimensional arrays.")
    return array[np.isfinite(array)]


def bootstrap_confidence_interval(
    data: Sequence[Sequence[float] | np.ndarray] | Sequence[float] | np.ndarray,
    statistic: Callable[..., float],
    *,
    paired: bool = False,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    method: str = "BCa",
    random_state: int = 0,
) -> BootstrapResult:
    """Estimate a scalar statistic and deterministic bootstrap confidence interval.

    The function intentionally returns only the compact interval contract. Ruddy does
    not persist every resampled statistic unless a later specialized workflow requires
    it.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100.")
    normalized_method = str(method).strip().lower()
    method_map = {"percentile": "percentile", "basic": "basic", "bca": "BCa"}
    if normalized_method not in method_map:
        raise ValueError("method must be one of percentile, basic, or bca.")

    if isinstance(data, np.ndarray) and data.ndim == 1:
        arrays = (_as_1d_finite(data),)
    elif not isinstance(data, np.ndarray) and data and np.isscalar(data[0]):  # type: ignore[index]
        arrays = (_as_1d_finite(data),)  # type: ignore[arg-type]
    else:
        arrays = tuple(_as_1d_finite(values) for values in data)  # type: ignore[arg-type]

    if not arrays or any(array.size < 2 for array in arrays):
        return BootstrapResult(
            estimate=float("nan"),
            confidence_low=float("nan"),
            confidence_high=float("nan"),
            standard_error=float("nan"),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method=normalized_method,
            random_state=random_state,
            status="skipped",
            reason="insufficient_observations",
        )
    if paired and len({array.size for array in arrays}) != 1:
        raise ValueError("Paired bootstrap inputs must have identical lengths.")

    try:
        estimate = float(statistic(*arrays))
    except Exception:
        estimate = float("nan")
    if not np.isfinite(estimate):
        return BootstrapResult(
            estimate=estimate,
            confidence_low=float("nan"),
            confidence_high=float("nan"),
            standard_error=float("nan"),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method=normalized_method,
            random_state=random_state,
            status="degenerate",
            reason="undefined_statistic",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.bootstrap(
                arrays,
                statistic,
                vectorized=False,
                paired=paired,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                method=method_map[normalized_method],
                rng=np.random.default_rng(random_state),
            )
    except Exception:
        return BootstrapResult(
            estimate=estimate,
            confidence_low=float("nan"),
            confidence_high=float("nan"),
            standard_error=float("nan"),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method=normalized_method,
            random_state=random_state,
            status="degenerate",
            reason="bootstrap_failed",
        )

    low = float(result.confidence_interval.low)
    high = float(result.confidence_interval.high)
    se = float(result.standard_error)
    if not all(np.isfinite(value) for value in (low, high, se)):
        return BootstrapResult(
            estimate=estimate,
            confidence_low=low,
            confidence_high=high,
            standard_error=se,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method=normalized_method,
            random_state=random_state,
            status="degenerate",
            reason="undefined_bootstrap_interval",
        )
    return BootstrapResult(
        estimate=estimate,
        confidence_low=low,
        confidence_high=high,
        standard_error=se,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        method=normalized_method,
        random_state=random_state,
    )
