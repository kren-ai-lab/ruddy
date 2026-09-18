"""Restricted mixed-effects modeling for exploratory hierarchical data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.formula.api import mixedlm

from ruddy.core.enums import ColumnKind, ResultStatus
from ruddy.factorial.design import FactorialDesign, build_factorial_design
from ruddy.projections.preprocessing import _exclusions_table
from ruddy.results import Advisory, AnalysisProvenance

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from ruddy.data import TabularDataset

FIXED_SCHEMA: dict[str, PolarsDataType] = {
    "parameter": pl.String,
    "estimate": pl.Float64,
    "std_error": pl.Float64,
    "z_value": pl.Float64,
    "p_value": pl.Float64,
    "ci_lower": pl.Float64,
    "ci_upper": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
FIXED_COLUMNS = tuple(FIXED_SCHEMA)

VARIANCE_SCHEMA: dict[str, PolarsDataType] = {
    "component": pl.String,
    "row": pl.String,
    "column": pl.String,
    "estimate": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
VARIANCE_COLUMNS = tuple(VARIANCE_SCHEMA)

RANDOM_EFFECT_SCHEMA: dict[str, PolarsDataType] = {
    "group": pl.String,
    "effect": pl.String,
    "estimate": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
RANDOM_EFFECT_COLUMNS = tuple(RANDOM_EFFECT_SCHEMA)


@dataclass(frozen=True, slots=True)
class MixedEffectsResult:
    """Structured random-intercept / numeric random-slope mixed-model result."""

    status: ResultStatus
    reason: str | None
    fixed_effects: pl.DataFrame
    variance_components: pl.DataFrame
    random_effects: pl.DataFrame
    exclusions: pl.DataFrame
    model_summary: dict[str, Any]
    advisories: tuple[Advisory, ...]
    provenance: AnalysisProvenance


def _safe_fixed_frame(
    model_frame: pl.DataFrame,
    design: FactorialDesign,
    group: str,
    random_slopes: tuple[str, ...],
) -> tuple[pd.DataFrame, str, dict[str, str], dict[str, str]]:
    # pandas boundary: statsmodels/Patsy consume pandas; see DEVELOPMENT.md
    frame = model_frame.to_pandas()
    safe = pd.DataFrame(index=frame.index)
    safe["Y"] = pd.to_numeric(frame[design.response], errors="raise").astype(float)
    safe["G"] = frame[group].astype("category")
    expression: dict[str, str] = {}
    random_names: dict[str, str] = {}
    for idx, name in enumerate(design.factors):
        safe_name = f"F{idx}"
        safe[safe_name] = frame[name].astype("category")
        expression[name] = f"C({safe_name}, Sum)"
    for idx, name in enumerate(design.covariates):
        safe_name = f"X{idx}"
        safe[safe_name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        expression[name] = safe_name
        if name in random_slopes:
            random_names[name] = safe_name
    rhs = " + ".join(":".join(expression[name] for name in term.columns) for term in design.terms)
    fixed_formula = f"Y ~ {rhs}"
    ("1" if not random_slopes else "1 + " + " + ".join(random_names[name] for name in random_slopes))
    return safe, fixed_formula, expression, random_names


def _finite_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def analyze_mixed_effects(
    dataset: TabularDataset,
    *,
    group: str,
    response: str | None = None,
    factors: tuple[str, ...] = (),
    covariates: tuple[str, ...] = (),
    interactions: tuple[tuple[str, ...], ...] = (),
    formula: str | None = None,
    random_slopes: tuple[str, ...] = (),
    reml: bool = True,
    optimizer: str = "lbfgs",
    max_iter: int = 1000,
    confidence_level: float = 0.95,
    min_groups: int = 3,
    min_group_n: int = 2,
    max_interaction_order: int = 3,
) -> MixedEffectsResult:
    """Fit a random-intercept model with optional numeric random slopes.

    Ruddy intentionally does not provide a general mixed-model DSL here. Random effects
    are one grouping factor, a random intercept, and optional numeric random slopes.
    """
    if group not in dataset.frame.columns:
        raise ValueError(f"Unknown mixed-effects grouping column: {group!r}.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    if min_groups < 2 or min_group_n < 1:
        raise ValueError("min_groups must be >=2 and min_group_n >=1.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    random_slopes = tuple(str(value) for value in random_slopes)
    if len(set(random_slopes)) != len(random_slopes):
        raise ValueError("random_slopes cannot contain duplicates.")

    design = build_factorial_design(
        dataset,
        response=response,
        factors=factors,
        covariates=covariates,
        interactions=interactions,
        formula=formula,
        ss_type=2,
        max_interaction_order=max_interaction_order,
    )
    unknown_slopes = [name for name in random_slopes if name not in design.covariates]
    if unknown_slopes:
        raise ValueError(f"Random slopes must be declared numeric covariates; invalid={unknown_slopes}.")
    for name in random_slopes:
        if dataset.kind_of(name) is not ColumnKind.NUMERIC:
            raise ValueError(f"Random slope {name!r} must be numeric.")

    selected = tuple(dict.fromkeys((design.response, *design.factors, *design.covariates, group)))
    frame = dataset.frame.select(selected)
    numeric_columns = (design.response, *design.covariates)
    complete = np.ones(frame.height, dtype=bool)
    for name in numeric_columns:
        values = frame.get_column(name).cast(pl.Float64).fill_null(float("nan")).to_numpy()
        complete &= np.isfinite(values)
    for name in (*design.factors, group):
        col = frame.get_column(name)
        mask = col.is_not_null()
        if col.dtype.is_float():
            mask &= ~col.is_nan()
        complete &= mask.to_numpy()
    model_frame = frame.filter(pl.Series(complete))
    excluded_rows = np.flatnonzero(~complete).astype(np.int64)
    exclusions = _exclusions_table(
        dataset.observation_ids,
        excluded_rows,
        stage="mixed_effects_complete_case",
        reason="missing_or_non_finite_model_value",
    )

    group_counts_df = model_frame.get_column(group).value_counts(name="__ruddy_count")
    group_counts = {row[group]: int(row["__ruddy_count"]) for row in group_counts_df.iter_rows(named=True)}
    provenance = AnalysisProvenance(
        analysis="mixed_effects",
        parameters={
            "response": design.response,
            "factors": design.factors,
            "covariates": design.covariates,
            "interactions": design.interactions,
            "group": group,
            "random_slopes": random_slopes,
            "random_intercept": True,
            "reml": bool(reml),
            "optimizer": optimizer,
            "max_iter": int(max_iter),
            "confidence_level": confidence_level,
        },
        input_summary={
            "n_observations": dataset.frame.height,
            "n_complete_case": int(complete.sum()),
            "n_excluded": int((~complete).sum()),
            "n_groups": len(group_counts),
        },
    )
    empty_fixed = pl.DataFrame(schema=FIXED_SCHEMA)
    empty_var = pl.DataFrame(schema=VARIANCE_SCHEMA)
    empty_random = pl.DataFrame(schema=RANDOM_EFFECT_SCHEMA)

    if len(group_counts) < min_groups:
        return MixedEffectsResult(
            ResultStatus.DEGENERATE,
            "insufficient_groups",
            empty_fixed,
            empty_var,
            empty_random,
            exclusions,
            {"n_groups": len(group_counts)},
            (),
            provenance,
        )
    if any(count < min_group_n for count in group_counts.values()):
        return MixedEffectsResult(
            ResultStatus.DEGENERATE,
            "group_too_small",
            empty_fixed,
            empty_var,
            empty_random,
            exclusions,
            {"group_counts": group_counts},
            (),
            provenance,
        )
    if model_frame.get_column(design.response).n_unique() < 2:
        return MixedEffectsResult(
            ResultStatus.DEGENERATE,
            "constant_response",
            empty_fixed,
            empty_var,
            empty_random,
            exclusions,
            {},
            (),
            provenance,
        )

    safe, fixed_formula, _, random_names = _safe_fixed_frame(model_frame, design, group, random_slopes)
    re_formula = (
        "1" if not random_slopes else "1 + " + " + ".join(random_names[name] for name in random_slopes)
    )
    advisories: list[Advisory] = []
    try:
        model = mixedlm(fixed_formula, safe, groups=safe["G"], re_formula=re_formula)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fit = model.fit(reml=reml, method=optimizer, maxiter=max_iter, disp=False)
        for warning in captured:
            advisories.append(
                Advisory(
                    code="mixed_model_fit_warning",
                    message=str(warning.message),
                    context={"warning_category": warning.category.__name__},
                )
            )
    except (ValueError, np.linalg.LinAlgError) as exc:
        advisories.append(Advisory(code="mixed_model_fit_error", message=str(exc)))
        return MixedEffectsResult(
            ResultStatus.DEGENERATE,
            "mixed_model_fit_failed",
            empty_fixed,
            empty_var,
            empty_random,
            exclusions,
            {"fixed_formula": fixed_formula, "re_formula": re_formula},
            tuple(advisories),
            provenance,
        )

    ci = fit.conf_int(alpha=1.0 - confidence_level)
    fixed_rows = []
    for name in fit.fe_params.index:
        estimate = float(fit.fe_params[name])
        se = float(fit.bse_fe[name])
        z_value = estimate / se if se > 0 else (np.inf if estimate != 0 else 0.0)
        p_val = fit.pvalues.get(name)
        p_value = _finite_or_none(p_val) if p_val is not None else None
        ci_low = _finite_or_none(ci.loc[name, 0])
        ci_high = _finite_or_none(ci.loc[name, 1])
        fixed_rows.append(
            {
                "parameter": str(name),
                "estimate": estimate,
                "std_error": se,
                "z_value": float(z_value),
                "p_value": p_value,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "status": ResultStatus.OK.value,
                "reason": None,
            }
        )

    variance_rows = [
        {
            "component": "residual",
            "row": "residual",
            "column": "residual",
            "estimate": float(fit.scale),
            "status": ResultStatus.OK.value,
            "reason": None,
        }
    ]
    cov_re = pd.DataFrame(fit.cov_re)
    for row_name in cov_re.index:
        for col_name in cov_re.columns:
            variance_rows.append(
                {
                    "component": "random_effect_covariance",
                    "row": str(row_name),
                    "column": str(col_name),
                    "estimate": float(cov_re.loc[row_name, col_name]),
                    "status": ResultStatus.OK.value,
                    "reason": None,
                }
            )

    random_rows = []
    try:
        for level, effects in fit.random_effects.items():
            for effect_name, estimate in pd.Series(effects).items():
                random_rows.append(
                    {
                        "group": str(level),
                        "effect": str(effect_name),
                        "estimate": float(estimate),
                        "status": ResultStatus.OK.value,
                        "reason": None,
                    }
                )
    except (ValueError, np.linalg.LinAlgError) as exc:
        advisories.append(
            Advisory(
                code="random_effects_unavailable",
                message="Conditional random-effect estimates could not be recovered.",
                context={"error": str(exc)},
            )
        )

    eigenvalues = np.linalg.eigvalsh(np.asarray(cov_re, dtype=float)) if cov_re.size else np.array([])
    singular = bool(eigenvalues.size and np.min(eigenvalues) <= 1e-10 * max(1.0, float(np.max(eigenvalues))))
    converged = bool(getattr(fit, "converged", False))
    if singular:
        advisories.append(
            Advisory(
                code="singular_random_effect_covariance",
                message="Estimated random-effect covariance is on or near the singular boundary.",
                context={"eigenvalues": eigenvalues.tolist()},
            )
        )
    if not converged:
        advisories.append(
            Advisory(
                code="mixed_model_not_converged",
                message="Mixed-effects optimizer did not report convergence.",
            )
        )

    random_intercept_var = float(cov_re.iloc[0, 0]) if cov_re.shape[0] else np.nan  # pyrefly: ignore[bad-argument-type]
    residual_var = float(fit.scale)
    icc = (
        random_intercept_var / (random_intercept_var + residual_var)
        if random_intercept_var >= 0 and residual_var >= 0 and (random_intercept_var + residual_var) > 0
        else np.nan
    )
    summary = {
        "fixed_formula": fixed_formula,
        "random_formula": re_formula,
        "n_complete_case": int(complete.sum()),
        "n_groups": len(group_counts),
        "converged": converged,
        "singular_random_effect_covariance": singular,
        "reml": bool(reml),
        "log_likelihood": _finite_or_none(fit.llf),
        "aic": _finite_or_none(fit.aic),
        "bic": _finite_or_none(fit.bic),
        "residual_variance": residual_var,
        "random_intercept_variance": random_intercept_var,
        "icc_random_intercept": _finite_or_none(icc),
    }
    status = ResultStatus.OK if converged else ResultStatus.DEGENERATE
    reason = None if converged else "mixed_model_not_converged"
    return MixedEffectsResult(
        status=status,
        reason=reason,
        fixed_effects=pl.DataFrame(fixed_rows, schema=FIXED_SCHEMA),
        variance_components=pl.DataFrame(variance_rows, schema=VARIANCE_SCHEMA),
        random_effects=pl.DataFrame(random_rows, schema=RANDOM_EFFECT_SCHEMA),
        exclusions=exclusions,
        model_summary=summary,
        advisories=tuple(advisories),
        provenance=provenance,
    )


__all__ = ["MixedEffectsResult", "analyze_mixed_effects"]
