"""Small analytic/sampling Bayesian EDA primitives with explicit weak priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from polars._typing import PolarsDataType
from scipy import stats

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.data import TabularDataset
from ruddy.results import AnalysisProvenance

MEANS_SCHEMA: dict[str, PolarsDataType] = {
    "variable": pl.String,
    "n": pl.Int64,
    "posterior_mean": pl.Float64,
    "posterior_median": pl.Float64,
    "credible_low": pl.Float64,
    "credible_high": pl.Float64,
    "probability_positive": pl.Float64,
    "probability_negative": pl.Float64,
    "probability_direction": pl.Float64,
    "rope_probability": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
MEANS_COLUMNS: tuple[str, ...] = tuple(MEANS_SCHEMA)

MEAN_DIFFERENCES_SCHEMA: dict[str, PolarsDataType] = {
    "variable": pl.String,
    "group": pl.String,
    "level_a": pl.String,
    "level_b": pl.String,
    "n_a": pl.Int64,
    "n_b": pl.Int64,
    "posterior_mean": pl.Float64,
    "posterior_median": pl.Float64,
    "credible_low": pl.Float64,
    "credible_high": pl.Float64,
    "probability_positive": pl.Float64,
    "probability_negative": pl.Float64,
    "probability_direction": pl.Float64,
    "rope_probability": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
MEAN_DIFFERENCES_COLUMNS: tuple[str, ...] = tuple(MEAN_DIFFERENCES_SCHEMA)

MEAN_DIFFERENCES_SCHEMA_BASE: dict[str, PolarsDataType] = {
    col: dtype for col, dtype in MEAN_DIFFERENCES_SCHEMA.items() if col not in {"level_a", "level_b"}
}


@dataclass(frozen=True, slots=True)
class BayesianEDAResult:
    means: pl.DataFrame
    mean_differences: pl.DataFrame
    provenance: AnalysisProvenance


def _posterior_parameters(
    values: np.ndarray,
    *,
    prior_mean: float,
    prior_kappa: float,
    prior_alpha: float,
    prior_beta: float,
) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean())
    ss = float(np.sum((x - mean) ** 2))
    kappa_n = prior_kappa + n
    mean_n = (prior_kappa * prior_mean + n * mean) / kappa_n
    alpha_n = prior_alpha + n / 2.0
    beta_n = prior_beta + 0.5 * ss + (prior_kappa * n * (mean - prior_mean) ** 2) / (2.0 * kappa_n)
    return mean_n, kappa_n, alpha_n, beta_n


def _sample_mean_posterior(
    values: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
    prior_mean: float,
    prior_kappa: float,
    prior_alpha: float,
    prior_beta: float,
) -> np.ndarray:
    mean_n, kappa_n, alpha_n, beta_n = _posterior_parameters(
        values,
        prior_mean=prior_mean,
        prior_kappa=prior_kappa,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )
    sigma2 = stats.invgamma(a=alpha_n, scale=beta_n).rvs(size=draws, random_state=rng)
    return rng.normal(mean_n, np.sqrt(sigma2 / kappa_n), size=draws)


def _posterior_summary(draws: np.ndarray, *, level: float, rope: tuple[float, float]) -> dict[str, float]:
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    p_pos = float(np.mean(draws > 0))
    p_neg = float(np.mean(draws < 0))
    return {
        "posterior_mean": float(np.mean(draws)),
        "posterior_median": float(np.median(draws)),
        "credible_low": float(low),
        "credible_high": float(high),
        "probability_positive": p_pos,
        "probability_negative": p_neg,
        "probability_direction": max(p_pos, p_neg),
        "rope_probability": float(np.mean((draws >= rope[0]) & (draws <= rope[1]))),
    }


def _build_mean_differences_frame(
    rows: list[dict[str, Any]],
    frame: pl.DataFrame,
    groups: tuple[str, ...],
) -> pl.DataFrame:
    group_dtype = frame.get_column(groups[0]).dtype if groups else pl.String
    if not rows:
        schema = {
            **MEAN_DIFFERENCES_SCHEMA_BASE,
            "level_a": group_dtype,
            "level_b": group_dtype,
        }
        return pl.DataFrame(schema={col: schema[col] for col in MEAN_DIFFERENCES_COLUMNS})

    has_any_level = any(row.get("level_a") is not None or row.get("level_b") is not None for row in rows)
    if not has_any_level:
        overrides = {
            **MEAN_DIFFERENCES_SCHEMA_BASE,
            "level_a": group_dtype,
            "level_b": group_dtype,
        }
        result_frame = pl.DataFrame(rows, schema_overrides=overrides)
    else:
        result_frame = pl.DataFrame(rows, schema_overrides=MEAN_DIFFERENCES_SCHEMA_BASE)
        if result_frame.get_column("level_a").dtype != group_dtype:
            result_frame = result_frame.with_columns(
                pl.col("level_a").cast(group_dtype),
                pl.col("level_b").cast(group_dtype),
            )
    for col in MEAN_DIFFERENCES_COLUMNS:
        if col not in result_frame.columns:
            dtype = group_dtype if col in {"level_a", "level_b"} else MEAN_DIFFERENCES_SCHEMA_BASE[col]
            result_frame = result_frame.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return result_frame.select(list(MEAN_DIFFERENCES_COLUMNS))


def analyze_bayesian_eda(
    dataset: TabularDataset,
    *,
    variables: tuple[str, ...] | None = None,
    groups: tuple[str, ...] = (),
    credible_level: float = 0.95,
    rope: tuple[float, float] = (-0.1, 0.1),
    draws: int = 5000,
    prior_mean: float = 0.0,
    prior_kappa: float = 1e-6,
    prior_alpha: float = 1e-6,
    prior_beta: float = 1e-6,
    min_n: int = 3,
    random_state: int = 0,
) -> BayesianEDAResult:
    """Estimate Bayesian means and binary-group mean differences under a weak N-IG prior."""
    if not 0 < credible_level < 1:
        raise ValueError("credible_level must lie strictly between 0 and 1.")
    if draws < 100:
        raise ValueError("draws must be at least 100.")
    if rope[0] > rope[1]:
        raise ValueError("rope lower bound cannot exceed upper bound.")
    frame = dataset.frame
    if variables is None:
        variables = tuple(
            c
            for c in frame.columns
            if dataset.kind_of(c) is ColumnKind.NUMERIC
            and dataset.role_of(c) in {ColumnRole.VARIABLE, ColumnRole.RESPONSE, ColumnRole.COVARIATE}
        )
    for column in variables:
        if column not in frame.columns or dataset.kind_of(column) is not ColumnKind.NUMERIC:
            raise ValueError(f"Bayesian EDA variable must be numeric: {column!r}.")
    rng = np.random.default_rng(random_state)
    mean_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    for column in variables:
        values = frame.get_column(column).cast(pl.Float64).fill_null(float("nan")).to_numpy()
        finite = values[np.isfinite(values)]
        if finite.size < min_n:
            mean_rows.append(
                {
                    "variable": column,
                    "n": int(finite.size),
                    "status": "skipped",
                    "reason": "insufficient_finite_observations",
                }
            )
            continue
        draws_mu = _sample_mean_posterior(
            finite,
            draws=draws,
            rng=rng,
            prior_mean=prior_mean,
            prior_kappa=prior_kappa,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )
        row = {
            "variable": column,
            "n": int(finite.size),
            **_posterior_summary(draws_mu, level=credible_level, rope=rope),
            "status": "ok",
            "reason": None,
        }
        mean_rows.append(row)
        for group in groups:
            if group not in frame.columns:
                raise ValueError(f"Unknown Bayesian grouping column: {group!r}.")
            g = frame.get_column(group)
            if g.dtype.is_float():
                non_missing = g.filter(~g.is_null() & ~g.is_nan())
            else:
                non_missing = g.drop_nulls()
            levels = list(dict.fromkeys(non_missing.to_list()))
            if len(levels) != 2:
                diff_rows.append(
                    {
                        "variable": column,
                        "group": group,
                        "level_a": None,
                        "level_b": None,
                        "n_a": None,
                        "n_b": None,
                        "posterior_mean": None,
                        "posterior_median": None,
                        "credible_low": None,
                        "credible_high": None,
                        "probability_positive": None,
                        "probability_negative": None,
                        "probability_direction": None,
                        "rope_probability": None,
                        "status": "skipped",
                        "reason": "bayesian_mean_difference_requires_two_levels",
                    }
                )
                continue
            a, b = levels
            g_list = g.to_list()
            mask_a = np.array([v == a for v in g_list])
            mask_b = np.array([v == b for v in g_list])
            va = values[mask_a]
            va = va[np.isfinite(va)]
            vb = values[mask_b]
            vb = vb[np.isfinite(vb)]
            if len(va) < min_n or len(vb) < min_n:
                diff_rows.append(
                    {
                        "variable": column,
                        "group": group,
                        "level_a": a,
                        "level_b": b,
                        "n_a": len(va),
                        "n_b": len(vb),
                        "posterior_mean": None,
                        "posterior_median": None,
                        "credible_low": None,
                        "credible_high": None,
                        "probability_positive": None,
                        "probability_negative": None,
                        "probability_direction": None,
                        "rope_probability": None,
                        "status": "skipped",
                        "reason": "group_too_small",
                    }
                )
                continue
            da = _sample_mean_posterior(
                va,
                draws=draws,
                rng=rng,
                prior_mean=prior_mean,
                prior_kappa=prior_kappa,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
            )
            db = _sample_mean_posterior(
                vb,
                draws=draws,
                rng=rng,
                prior_mean=prior_mean,
                prior_kappa=prior_kappa,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
            )
            dd = da - db
            diff_rows.append(
                {
                    "variable": column,
                    "group": group,
                    "level_a": a,
                    "level_b": b,
                    "n_a": len(va),
                    "n_b": len(vb),
                    **_posterior_summary(dd, level=credible_level, rope=rope),
                    "status": "ok",
                    "reason": None,
                }
            )
    provenance = AnalysisProvenance(
        analysis="bayesian_eda",
        parameters={
            "credible_level": credible_level,
            "rope": rope,
            "draws": draws,
            "prior": {"mean": prior_mean, "kappa": prior_kappa, "alpha": prior_alpha, "beta": prior_beta},
            "groups": groups,
            "min_n": min_n,
        },
        input_summary={"n_observations": frame.height, "variables": variables},
        random_state=random_state,
    )
    mean_frame = pl.DataFrame(mean_rows, schema=MEANS_SCHEMA)
    diff_frame = _build_mean_differences_frame(diff_rows, frame, groups)
    return BayesianEDAResult(mean_frame, diff_frame, provenance)


__all__ = ["BayesianEDAResult", "analyze_bayesian_eda"]
