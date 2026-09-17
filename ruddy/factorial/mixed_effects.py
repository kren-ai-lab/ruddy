"""Restricted mixed-effects modeling for exploratory hierarchical data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm

from ruddy.core.enums import ColumnKind, ResultStatus
from ruddy.factorial.design import FactorialDesign, build_factorial_design
from ruddy.results import Advisory, AnalysisProvenance

if TYPE_CHECKING:
    from ruddy.data import TabularDataset

FIXED_COLUMNS = (
    "parameter",
    "estimate",
    "std_error",
    "z_value",
    "p_value",
    "ci_lower",
    "ci_upper",
    "status",
    "reason",
)
VARIANCE_COLUMNS = ("component", "row", "column", "estimate", "status", "reason")
RANDOM_EFFECT_COLUMNS = ("group", "effect", "estimate", "status", "reason")
EXCLUSION_COLUMNS = ("source_row_index", "observation_id", "stage", "reason")


@dataclass(frozen=True, slots=True)
class MixedEffectsResult:
    """Structured random-intercept / numeric random-slope mixed-model result."""

    status: ResultStatus
    reason: str | None
    fixed_effects: pd.DataFrame
    variance_components: pd.DataFrame
    random_effects: pd.DataFrame
    exclusions: pd.DataFrame
    model_summary: dict[str, Any]
    advisories: tuple[Advisory, ...]
    provenance: AnalysisProvenance


def _safe_fixed_frame(
    frame: pd.DataFrame,
    design: FactorialDesign,
    group: str,
    random_slopes: tuple[str, ...],
) -> tuple[pd.DataFrame, str, dict[str, str], dict[str, str]]:
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


def _finite_or_none(value: Any) -> float | None:
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
    if group not in dataset.columns:
        msg = f"Unknown mixed-effects grouping column: {group!r}."
        raise ValueError(msg)
    if not 0.0 < confidence_level < 1.0:
        msg = "confidence_level must lie in (0, 1)."
        raise ValueError(msg)
    if min_groups < 2 or min_group_n < 1:
        msg = "min_groups must be >=2 and min_group_n >=1."
        raise ValueError(msg)
    if max_iter < 1:
        msg = "max_iter must be positive."
        raise ValueError(msg)
    random_slopes = tuple(str(value) for value in random_slopes)
    if len(set(random_slopes)) != len(random_slopes):
        msg = "random_slopes cannot contain duplicates."
        raise ValueError(msg)

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
        msg = f"Random slopes must be declared numeric covariates; invalid={unknown_slopes}."
        raise ValueError(msg)
    for name in random_slopes:
        if dataset.kind_of(name) is not ColumnKind.NUMERIC:
            msg = f"Random slope {name!r} must be numeric."
            raise ValueError(msg)

    selected = tuple(dict.fromkeys((design.response, *design.factors, *design.covariates, group)))
    frame = dataset.select(selected)
    complete = np.ones(len(frame), dtype=bool)
    for name in (design.response, *design.covariates):
        complete &= np.isfinite(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
    for name in (*design.factors, group):
        complete &= frame[name].notna().to_numpy()
    model_frame = frame.loc[complete].reset_index(drop=True)
    excluded_idx = np.flatnonzero(~complete)
    exclusions = pd.DataFrame(
        {
            "source_row_index": excluded_idx.astype(np.int64),
            "observation_id": dataset.observation_ids.take(excluded_idx).to_list(),
            "stage": "mixed_effects_complete_case",
            "reason": "missing_or_non_finite_model_value",
        },
        columns=pd.Index(EXCLUSION_COLUMNS),
    )

    group_counts = model_frame[group].value_counts(dropna=False)
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
            "n_observations": dataset.n_observations,
            "n_complete_case": int(complete.sum()),
            "n_excluded": int((~complete).sum()),
            "n_groups": len(group_counts),
        },
    )
    empty_fixed = pd.DataFrame(columns=pd.Index(FIXED_COLUMNS))
    empty_var = pd.DataFrame(columns=pd.Index(VARIANCE_COLUMNS))
    empty_random = pd.DataFrame(columns=pd.Index(RANDOM_EFFECT_COLUMNS))

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
    if (group_counts < min_group_n).any():
        return MixedEffectsResult(
            ResultStatus.DEGENERATE,
            "group_too_small",
            empty_fixed,
            empty_var,
            empty_random,
            exclusions,
            {"group_counts": group_counts.to_dict()},
            (),
            provenance,
        )
    if model_frame[design.response].nunique(dropna=True) < 2:
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
        advisories.extend(
            Advisory(
                code="mixed_model_fit_warning",
                message=str(warning.message),
                context={"warning_category": warning.category.__name__},
            )
            for warning in captured
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
        p_value = float(fit.pvalues[name]) if name in fit.pvalues.index else np.nan
        fixed_rows.append(
            {
                "parameter": str(name),
                "estimate": estimate,
                "std_error": se,
                "z_value": float(z_value),
                "p_value": p_value,
                "ci_lower": float(ci.loc[name, 0]),
                "ci_upper": float(ci.loc[name, 1]),
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
    variance_rows.extend(
        {
            "component": "random_effect_covariance",
            "row": str(row_name),
            "column": str(col_name),
            "estimate": float(cov_re.loc[row_name, col_name]),
            "status": ResultStatus.OK.value,
            "reason": None,
        }
        for row_name in cov_re.index
        for col_name in cov_re.columns
    )

    random_rows = []
    try:
        for level, effects in fit.random_effects.items():
            for effect_name, estimate in pd.Series(effects).items():
                random_rows.append(
                    {
                        "group": level,
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
        fixed_effects=pd.DataFrame(fixed_rows, columns=pd.Index(FIXED_COLUMNS)),
        variance_components=pd.DataFrame(variance_rows, columns=pd.Index(VARIANCE_COLUMNS)),
        random_effects=pd.DataFrame(random_rows, columns=pd.Index(RANDOM_EFFECT_COLUMNS)),
        exclusions=exclusions,
        model_summary=summary,
        advisories=tuple(advisories),
        provenance=provenance,
    )


__all__ = ["MixedEffectsResult", "analyze_mixed_effects"]
