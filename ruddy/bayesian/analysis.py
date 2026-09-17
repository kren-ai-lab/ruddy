"""Small analytic/sampling Bayesian EDA primitives with explicit weak priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.results import AnalysisProvenance

if TYPE_CHECKING:
    from ruddy.data import TabularDataset


@dataclass(frozen=True, slots=True)
class BayesianEDAResult:
    means: pd.DataFrame
    mean_differences: pd.DataFrame
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
        msg = "credible_level must lie strictly between 0 and 1."
        raise ValueError(msg)
    if draws < 100:
        msg = "draws must be at least 100."
        raise ValueError(msg)
    if rope[0] > rope[1]:
        msg = "rope lower bound cannot exceed upper bound."
        raise ValueError(msg)
    frame = dataset.to_frame()
    if variables is None:
        variables = tuple(
            c
            for c in dataset.columns
            if dataset.kind_of(c) is ColumnKind.NUMERIC
            and dataset.role_of(c) in {ColumnRole.VARIABLE, ColumnRole.RESPONSE, ColumnRole.COVARIATE}
        )
    for column in variables:
        if column not in dataset.columns or dataset.kind_of(column) is not ColumnKind.NUMERIC:
            msg = f"Bayesian EDA variable must be numeric: {column!r}."
            raise ValueError(msg)
    rng = np.random.default_rng(random_state)
    mean_rows: list[dict] = []
    diff_rows: list[dict] = []
    for column in variables:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
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
            if group not in dataset.columns:
                msg = f"Unknown Bayesian grouping column: {group!r}."
                raise ValueError(msg)
            g = frame[group]
            levels = list(pd.unique(g.dropna()))
            if len(levels) != 2:
                diff_rows.append(
                    {
                        "variable": column,
                        "group": group,
                        "level_a": None,
                        "level_b": None,
                        "status": "skipped",
                        "reason": "bayesian_mean_difference_requires_two_levels",
                    }
                )
                continue
            a, b = levels
            va = pd.to_numeric(frame.loc[g == a, column], errors="raise").to_numpy(dtype=float)
            va = va[np.isfinite(va)]
            vb = pd.to_numeric(frame.loc[g == b, column], errors="raise").to_numpy(dtype=float)
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
        input_summary={"n_observations": dataset.n_observations, "variables": variables},
        random_state=random_state,
    )
    return BayesianEDAResult(pd.DataFrame(mean_rows), pd.DataFrame(diff_rows), provenance)


__all__ = ["BayesianEDAResult", "analyze_bayesian_eda"]
