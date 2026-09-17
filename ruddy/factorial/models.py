"""Factorial ANOVA/ANCOVA model fitting and structured results."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from ruddy.core.enums import PAdjustMethod, ResultStatus
from ruddy.factorial.design import FactorialDesign, FactorialTerm, build_factorial_design
from ruddy.factorial.diagnostics import _finite_or_none, build_factorial_cells, model_diagnostics
from ruddy.factorial.effects import factorial_effect_sizes
from ruddy.results import Advisory, AnalysisProvenance
from ruddy.statistics.multiple_testing import adjust_pvalues

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ruddy.data import TabularDataset

EFFECT_COLUMNS = (
    "term",
    "term_type",
    "order",
    "components_json",
    "df",
    "sum_sq",
    "mean_sq",
    "f_value",
    "p_value",
    "q_value",
    "correction",
    "family_id",
    "family_size",
    "eta_squared",
    "partial_eta_squared",
    "omega_squared",
    "partial_omega_squared",
    "status",
    "reason",
)

COEFFICIENT_COLUMNS = (
    "parameter",
    "estimate",
    "std_error",
    "t_value",
    "p_value",
    "ci_lower",
    "ci_upper",
    "status",
    "reason",
)

EXCLUSION_COLUMNS = (
    "source_row_index",
    "observation_id",
    "stage",
    "reason",
)

DESIGN_TERM_COLUMNS = (
    "term",
    "term_type",
    "order",
    "components_json",
    "component_kinds_json",
)


@dataclass(frozen=True, slots=True)
class FactorialResult:
    """Structured result for one univariate factorial ANOVA/ANCOVA model."""

    status: ResultStatus
    reason: str | None
    design: FactorialDesign
    design_terms: pd.DataFrame
    effects: pd.DataFrame
    coefficients: pd.DataFrame
    diagnostics: pd.DataFrame
    observation_diagnostics: pd.DataFrame
    cells: pd.DataFrame
    exclusions: pd.DataFrame
    model_summary: dict[str, Any]
    advisories: tuple[Advisory, ...]
    provenance: AnalysisProvenance


def _normalize_p_adjust(value: PAdjustMethod | str) -> PAdjustMethod:
    return value if isinstance(value, PAdjustMethod) else PAdjustMethod(value)


def _normalize_robust_covariance(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "nonrobust"}:
        return None
    if normalized not in {"hc0", "hc1", "hc2", "hc3"}:
        msg = "robust_covariance must be one of none, hc0, hc1, hc2, hc3."
        raise ValueError(msg)
    return normalized


def _design_term_table(design: FactorialDesign) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "term": term.label,
                "term_type": term.term_type,
                "order": term.order,
                "components_json": json.dumps(term.columns, separators=(",", ":")),
                "component_kinds_json": json.dumps(term.kinds, separators=(",", ":")),
            }
            for term in design.terms
        ],
        columns=pd.Index(DESIGN_TERM_COLUMNS),
    )


def _empty_result(
    *,
    status: ResultStatus,
    reason: str,
    design: FactorialDesign,
    design_terms: pd.DataFrame,
    cells: pd.DataFrame,
    exclusions: pd.DataFrame,
    model_summary: dict[str, Any],
    advisories: tuple[Advisory, ...],
    provenance: AnalysisProvenance,
) -> FactorialResult:
    from ruddy.factorial.diagnostics import DIAGNOSTIC_COLUMNS, OBSERVATION_DIAGNOSTIC_COLUMNS

    return FactorialResult(
        status=status,
        reason=reason,
        design=design,
        design_terms=design_terms,
        effects=pd.DataFrame(columns=pd.Index(EFFECT_COLUMNS)),
        coefficients=pd.DataFrame(columns=pd.Index(COEFFICIENT_COLUMNS)),
        diagnostics=pd.DataFrame(columns=pd.Index(DIAGNOSTIC_COLUMNS)),
        observation_diagnostics=pd.DataFrame(columns=pd.Index(OBSERVATION_DIAGNOSTIC_COLUMNS)),
        cells=cells,
        exclusions=exclusions,
        model_summary=model_summary,
        advisories=advisories,
        provenance=provenance,
    )


def _safe_model_frame(
    model_frame: pd.DataFrame,
    design: FactorialDesign,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict[str, str]]:
    safe = pd.DataFrame(index=model_frame.index)
    safe["Y"] = pd.to_numeric(model_frame[design.response], errors="raise").astype(float)
    factor_map: dict[str, str] = {}
    covariate_map: dict[str, str] = {}
    safe_expression: dict[str, str] = {}
    for index, name in enumerate(design.factors):
        safe_name = f"F{index}"
        safe[safe_name] = model_frame[name].astype("category")
        expression = f"C({safe_name}, Sum)"
        factor_map[safe_name] = name
        safe_expression[name] = expression
    for index, name in enumerate(design.covariates):
        safe_name = f"X{index}"
        safe[safe_name] = pd.to_numeric(model_frame[name], errors="raise").astype(float)
        covariate_map[safe_name] = name
        safe_expression[name] = safe_name
    return safe, factor_map, covariate_map, safe_expression


def _safe_term_expression(term: FactorialTerm, safe_expression: dict[str, str]) -> str:
    return ":".join(safe_expression[name] for name in term.columns)


def _translate_parameter(
    parameter: str,
    *,
    factor_map: dict[str, str],
    covariate_map: dict[str, str],
) -> str:
    translated = str(parameter)
    for safe_name, original in factor_map.items():
        translated = translated.replace(f"C({safe_name}, Sum)", original)
    for safe_name, original in covariate_map.items():
        translated = translated.replace(safe_name, original)
    return translated


def _make_provenance(
    dataset: TabularDataset,
    design: FactorialDesign,
    *,
    n_complete: int,
    n_excluded: int,
    p_adjust: PAdjustMethod,
    robust_covariance: str | None,
    min_cell_n: int,
    max_factor_levels: int,
    max_design_cells: int,
    max_design_columns: int,
    max_interaction_order: int,
    diagnostic_alpha: float,
) -> AnalysisProvenance:
    return AnalysisProvenance(
        analysis="factorial",
        parameters={
            "response": design.response,
            "factors": design.factors,
            "covariates": design.covariates,
            "interactions": design.interactions,
            "requested_formula": design.requested_formula,
            "resolved_formula": design.resolved_formula,
            "ss_type": design.ss_type,
            "factor_contrasts": "sum_to_zero",
            "p_adjust": p_adjust.value,
            "robust_covariance": robust_covariance,
            "min_cell_n": int(min_cell_n),
            "max_factor_levels": int(max_factor_levels),
            "max_design_cells": int(max_design_cells),
            "max_design_columns": int(max_design_columns),
            "max_interaction_order": int(max_interaction_order),
            "diagnostic_alpha": float(diagnostic_alpha),
        },
        input_summary={
            "n_observations": dataset.n_observations,
            "n_complete_case": int(n_complete),
            "n_excluded": int(n_excluded),
        },
    )


def analyze_factorial(
    dataset: TabularDataset,
    *,
    response: str | None = None,
    factors: Iterable[str] = (),
    covariates: Iterable[str] = (),
    interactions: Iterable[Iterable[str]] = (),
    formula: str | None = None,
    ss_type: int | str = 2,
    p_adjust: PAdjustMethod | str = PAdjustMethod.NONE,
    robust_covariance: str | None = None,
    min_cell_n: int = 2,
    max_factor_levels: int = 20,
    max_design_cells: int = 5000,
    max_design_columns: int = 500,
    max_interaction_order: int = 3,
    diagnostic_alpha: float = 0.05,
    condition_number_threshold: float = 30.0,
) -> FactorialResult:
    """Fit an explicit OLS factorial ANOVA/ANCOVA model.

    Ruddy does not choose sum-of-squares type, robust covariance, or model terms from
    assumption tests. Diagnostics are reported separately and never mutate the model.
    Type III models use sum-to-zero factor contrasts; the same contrast policy is used
    for Type II so changing SS type does not silently change the design coding.
    """
    if min_cell_n < 1:
        msg = "min_cell_n must be at least 1."
        raise ValueError(msg)
    if max_factor_levels < 2:
        msg = "max_factor_levels must be at least 2."
        raise ValueError(msg)
    if max_design_cells < 1:
        msg = "max_design_cells must be at least 1."
        raise ValueError(msg)
    if max_design_columns < 2:
        msg = "max_design_columns must be at least 2."
        raise ValueError(msg)
    if not 0.0 < diagnostic_alpha < 1.0:
        msg = "diagnostic_alpha must lie in (0, 1)."
        raise ValueError(msg)
    if condition_number_threshold <= 0.0:
        msg = "condition_number_threshold must be greater than zero."
        raise ValueError(msg)

    correction = _normalize_p_adjust(p_adjust)
    robust = _normalize_robust_covariance(robust_covariance)
    design = build_factorial_design(
        dataset,
        response=response,
        factors=factors,
        covariates=covariates,
        interactions=interactions,
        formula=formula,
        ss_type=ss_type,
        max_interaction_order=max_interaction_order,
    )
    design_terms = _design_term_table(design)

    selected = (design.response, *design.factors, *design.covariates)
    frame = dataset.select(selected)
    complete = np.ones(len(frame), dtype=bool)
    numeric_columns = (design.response, *design.covariates)
    for name in numeric_columns:
        numeric = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        complete &= np.isfinite(numeric)
    for factor in design.factors:
        complete &= frame[factor].notna().to_numpy()

    source_rows = np.flatnonzero(complete)
    excluded_rows = np.flatnonzero(~complete)
    ids = dataset.observation_ids
    exclusions = pd.DataFrame(
        {
            "source_row_index": excluded_rows.astype(np.int64),
            "observation_id": ids.take(excluded_rows).to_list(),
            "stage": "factorial_complete_case",
            "reason": "missing_or_non_finite_model_value",
        },
        columns=pd.Index(EXCLUSION_COLUMNS),
    )
    model_frame = frame.loc[complete].reset_index(drop=True)
    n = len(model_frame)

    provenance = _make_provenance(
        dataset,
        design,
        n_complete=n,
        n_excluded=len(excluded_rows),
        p_adjust=correction,
        robust_covariance=robust,
        min_cell_n=min_cell_n,
        max_factor_levels=max_factor_levels,
        max_design_cells=max_design_cells,
        max_design_columns=max_design_columns,
        max_interaction_order=max_interaction_order,
        diagnostic_alpha=diagnostic_alpha,
    )
    advisories: list[Advisory] = []

    for factor in design.factors:
        counts = model_frame[factor].value_counts(dropna=False, sort=False)
        if len(counts) < 2:
            return _empty_result(
                status=ResultStatus.DEGENERATE,
                reason="factor_has_fewer_than_two_levels",
                design=design,
                design_terms=design_terms,
                cells=pd.DataFrame(),
                exclusions=exclusions,
                model_summary={"n_complete_case": n, "factor": factor, "n_levels": len(counts)},
                advisories=tuple(advisories),
                provenance=provenance,
            )
        if len(counts) > max_factor_levels:
            msg = f"Factor {factor!r} has {len(counts)} levels; maximum is {max_factor_levels}."
            raise ValueError(msg)

    cells, cell_summary = build_factorial_cells(
        model_frame,
        design.factors,
        min_cell_n=min_cell_n,
        max_design_cells=max_design_cells,
    )
    if cell_summary["n_empty_cells"]:
        advisories.append(
            Advisory(
                code="empty_factorial_cells",
                message="One or more combinations of observed factor levels have no complete-case observations.",
                context={"n_empty_cells": cell_summary["n_empty_cells"]},
            )
        )
    if cell_summary["n_small_cells"]:
        advisories.append(
            Advisory(
                code="small_factorial_cells",
                message="One or more factorial cells contain fewer observations than min_cell_n.",
                context={"n_small_cells": cell_summary["n_small_cells"], "min_cell_n": min_cell_n},
            )
        )
    if design.factors and not cell_summary["balanced"]:
        advisories.append(
            Advisory(
                code="unbalanced_factorial_design",
                message="Factorial cell counts are unbalanced; interpret adjusted sums of squares accordingly.",
                context=cell_summary,
            )
        )
    if design.ss_type == 2 and design.interactions:
        advisories.append(
            Advisory(
                code="type_ii_with_interactions",
                message="Type II tests with interactions require careful interpretation of lower-order effects.",
                context={"interactions": design.interactions},
            )
        )

    if n == 0:
        return _empty_result(
            status=ResultStatus.SKIPPED,
            reason="no_complete_case_observations",
            design=design,
            design_terms=design_terms,
            cells=cells,
            exclusions=exclusions,
            model_summary={"n_complete_case": 0, **cell_summary},
            advisories=tuple(advisories),
            provenance=provenance,
        )
    response_values = model_frame[design.response].to_numpy(dtype=float)
    corrected_total_ss = float(np.sum((response_values - response_values.mean()) ** 2))
    if not math.isfinite(corrected_total_ss) or corrected_total_ss <= 0.0:
        return _empty_result(
            status=ResultStatus.DEGENERATE,
            reason="constant_response",
            design=design,
            design_terms=design_terms,
            cells=cells,
            exclusions=exclusions,
            model_summary={"n_complete_case": n, "corrected_total_ss": corrected_total_ss, **cell_summary},
            advisories=tuple(advisories),
            provenance=provenance,
        )

    safe, factor_map, covariate_map, safe_expression = _safe_model_frame(model_frame, design)
    safe_terms = [_safe_term_expression(term, safe_expression) for term in design.terms]
    safe_formula = "Y ~ " + " + ".join(safe_terms)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ols(safe_formula, data=safe).fit()
    exog = np.asarray(model.model.exog, dtype=float)
    design_rank = int(np.linalg.matrix_rank(exog))
    n_design_columns = int(exog.shape[1])
    residual_df = float(model.df_resid)
    if n_design_columns > max_design_columns:
        msg = f"Factorial design has {n_design_columns} columns; maximum is {max_design_columns}."
        raise ValueError(msg)
    if design_rank < n_design_columns:
        return _empty_result(
            status=ResultStatus.DEGENERATE,
            reason="rank_deficient_design",
            design=design,
            design_terms=design_terms,
            cells=cells,
            exclusions=exclusions,
            model_summary={
                "n_complete_case": n,
                "design_rank": design_rank,
                "n_design_columns": n_design_columns,
                "residual_df": residual_df,
                "safe_formula": safe_formula,
                **cell_summary,
            },
            advisories=tuple(advisories),
            provenance=provenance,
        )
    if residual_df <= 0.0:
        return _empty_result(
            status=ResultStatus.SKIPPED,
            reason="insufficient_residual_degrees_of_freedom",
            design=design,
            design_terms=design_terms,
            cells=cells,
            exclusions=exclusions,
            model_summary={
                "n_complete_case": n,
                "design_rank": design_rank,
                "n_design_columns": n_design_columns,
                "residual_df": residual_df,
                "safe_formula": safe_formula,
                **cell_summary,
            },
            advisories=tuple(advisories),
            provenance=provenance,
        )

    for caught_warning in caught:
        advisories.append(
            Advisory(
                code="model_fit_warning",
                message=str(caught_warning.message),
                context={"warning_class": caught_warning.category.__name__},
            )
        )

    with warnings.catch_warnings(record=True) as anova_warnings:
        warnings.simplefilter("always")
        anova = anova_lm(model, typ=design.ss_type, robust=robust)
    for caught_warning in anova_warnings:
        advisories.append(
            Advisory(
                code="anova_warning",
                message=str(caught_warning.message),
                context={"warning_class": caught_warning.category.__name__},
            )
        )

    residual_ss = float(np.sum(np.asarray(model.resid, dtype=float) ** 2))
    mse = residual_ss / residual_df
    safe_to_original = {_safe_term_expression(term, safe_expression): term for term in design.terms}
    family_id = f"factorial:{design.response}:type{design.ss_type}"
    effect_rows: list[dict[str, Any]] = []
    for safe_term, term in safe_to_original.items():
        if safe_term not in anova.index:
            effect_rows.append(
                {
                    "term": term.label,
                    "term_type": term.term_type,
                    "order": term.order,
                    "components_json": json.dumps(term.columns, separators=(",", ":")),
                    "df": None,
                    "sum_sq": None,
                    "mean_sq": None,
                    "f_value": None,
                    "p_value": None,
                    "q_value": None,
                    "correction": correction.value,
                    "family_id": family_id,
                    "family_size": 0,
                    "eta_squared": None,
                    "partial_eta_squared": None,
                    "omega_squared": None,
                    "partial_omega_squared": None,
                    "status": ResultStatus.DEGENERATE.value,
                    "reason": "term_not_estimable",
                }
            )
            continue
        row = anova.loc[safe_term]
        df = _finite_or_none(row.get("df"))
        sum_sq = _finite_or_none(row.get("sum_sq"))
        f_value = _finite_or_none(row.get("F"))
        p_value = _finite_or_none(row.get("PR(>F)"))
        status = ResultStatus.OK
        reason = None
        if df is None or sum_sq is None or f_value is None or p_value is None or df <= 0.0:
            status = ResultStatus.DEGENERATE
            reason = "non_finite_term_inference"
            sizes = {
                "eta_squared": None,
                "partial_eta_squared": None,
                "omega_squared": None,
                "partial_omega_squared": None,
            }
        else:
            sizes = factorial_effect_sizes(
                effect_ss=sum_sq,
                effect_df=df,
                residual_ss=residual_ss,
                residual_df=residual_df,
                corrected_total_ss=corrected_total_ss,
            )
        effect_rows.append(
            {
                "term": term.label,
                "term_type": term.term_type,
                "order": term.order,
                "components_json": json.dumps(term.columns, separators=(",", ":")),
                "df": df,
                "sum_sq": sum_sq,
                "mean_sq": None if df is None or sum_sq is None or df <= 0 else sum_sq / df,
                "f_value": f_value,
                "p_value": p_value,
                "q_value": None,
                "correction": correction.value,
                "family_id": family_id,
                "family_size": 0,
                **sizes,
                "status": status.value,
                "reason": reason,
            }
        )
    effects = pd.DataFrame(effect_rows, columns=pd.Index(EFFECT_COLUMNS))
    inferential = effects["status"].eq(ResultStatus.OK.value)
    family_size = int(inferential.sum())
    effects.loc[:, "family_size"] = family_size
    if family_size:
        p_values = effects.loc[inferential, "p_value"].to_numpy(dtype=float)
        effects.loc[inferential, "q_value"] = adjust_pvalues(p_values, correction)

    ci = model.conf_int(alpha=diagnostic_alpha)
    coefficient_rows: list[dict[str, Any]] = []
    for parameter in model.params.index:
        estimate = _finite_or_none(model.params[parameter])
        std_error = _finite_or_none(model.bse[parameter])
        t_value = _finite_or_none(model.tvalues[parameter])
        p_value = _finite_or_none(model.pvalues[parameter])
        ci_lower = _finite_or_none(ci.loc[parameter, 0])
        ci_upper = _finite_or_none(ci.loc[parameter, 1])
        ok = all(value is not None for value in (estimate, std_error, t_value, p_value, ci_lower, ci_upper))
        coefficient_rows.append(
            {
                "parameter": _translate_parameter(
                    parameter, factor_map=factor_map, covariate_map=covariate_map
                ),
                "estimate": estimate,
                "std_error": std_error,
                "t_value": t_value,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "status": ResultStatus.OK.value if ok else ResultStatus.DEGENERATE.value,
                "reason": None if ok else "non_finite_coefficient_inference",
            }
        )
    coefficients = pd.DataFrame(coefficient_rows, columns=pd.Index(COEFFICIENT_COLUMNS))

    diagnostics, observation_diagnostics = model_diagnostics(
        model,
        model_frame=model_frame,
        source_row_indices=source_rows,
        observation_ids=ids,
        factors=design.factors,
        alpha=diagnostic_alpha,
        condition_number_threshold=condition_number_threshold,
    )
    flagged = diagnostics.loc[
        diagnostics["status"].eq(ResultStatus.OK.value) & diagnostics["flagged"].eq(True),
        "diagnostic",
    ].tolist()
    if flagged:
        advisories.append(
            Advisory(
                code="factorial_diagnostics_flagged",
                message="One or more model diagnostics crossed their configured reference threshold.",
                context={"diagnostics": tuple(flagged), "alpha": diagnostic_alpha},
            )
        )
    if not observation_diagnostics.empty:
        n_influential = int(observation_diagnostics["is_influential"].fillna(False).sum())
        n_high_leverage = int(observation_diagnostics["is_high_leverage"].fillna(False).sum())
        n_large_residual = int(observation_diagnostics["is_large_residual"].fillna(False).sum())
        if n_influential or n_high_leverage or n_large_residual:
            advisories.append(
                Advisory(
                    code="influence_diagnostics_flagged",
                    message="Observation-level influence diagnostics flagged one or more complete-case observations.",
                    context={
                        "n_influential": n_influential,
                        "n_high_leverage": n_high_leverage,
                        "n_large_residual": n_large_residual,
                    },
                )
            )

    model_summary = {
        "response": design.response,
        "factors": design.factors,
        "covariates": design.covariates,
        "interactions": design.interactions,
        "requested_formula": design.requested_formula,
        "resolved_formula": design.resolved_formula,
        "safe_formula": safe_formula,
        "ss_type": design.ss_type,
        "factor_contrasts": "sum_to_zero",
        "robust_covariance": robust,
        "n_observations": dataset.n_observations,
        "n_complete_case": n,
        "n_excluded": len(excluded_rows),
        "n_design_columns": n_design_columns,
        "design_rank": design_rank,
        "residual_df": residual_df,
        "residual_ss": residual_ss,
        "residual_mse": mse,
        "corrected_total_ss": corrected_total_ss,
        "r_squared": _finite_or_none(model.rsquared),
        "adjusted_r_squared": _finite_or_none(model.rsquared_adj),
        "aic": _finite_or_none(model.aic),
        "bic": _finite_or_none(model.bic),
        **cell_summary,
    }
    return FactorialResult(
        status=ResultStatus.OK,
        reason=None,
        design=design,
        design_terms=design_terms,
        effects=effects,
        coefficients=coefficients,
        diagnostics=diagnostics,
        observation_diagnostics=observation_diagnostics,
        cells=cells,
        exclusions=exclusions,
        model_summary=model_summary,
        advisories=tuple(advisories),
        provenance=provenance,
    )


__all__ = [
    "COEFFICIENT_COLUMNS",
    "DESIGN_TERM_COLUMNS",
    "EFFECT_COLUMNS",
    "EXCLUSION_COLUMNS",
    "FactorialResult",
    "analyze_factorial",
]
