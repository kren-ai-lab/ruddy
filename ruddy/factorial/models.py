"""Factorial ANOVA/ANCOVA model fitting and structured results."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from polars._typing import PolarsDataType
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from ruddy.core.enums import PAdjustMethod, ResultStatus
from ruddy.data import TabularDataset
from ruddy.factorial.design import FactorialDesign, FactorialTerm, build_factorial_design
from ruddy.factorial.diagnostics import _finite_or_none, build_factorial_cells, model_diagnostics
from ruddy.factorial.effects import factorial_effect_sizes
from ruddy.projections.preprocessing import EXCLUSION_COLUMNS, _exclusions_table
from ruddy.results import Advisory, AnalysisProvenance
from ruddy.statistics.multiple_testing import adjust_pvalues

EFFECT_SCHEMA: dict[str, PolarsDataType] = {
    "term": pl.String,
    "term_type": pl.String,
    "order": pl.Int64,
    "components_json": pl.String,
    "df": pl.Float64,
    "sum_sq": pl.Float64,
    "mean_sq": pl.Float64,
    "f_value": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "correction": pl.String,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "eta_squared": pl.Float64,
    "partial_eta_squared": pl.Float64,
    "omega_squared": pl.Float64,
    "partial_omega_squared": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
EFFECT_COLUMNS = tuple(EFFECT_SCHEMA)

COEFFICIENT_SCHEMA: dict[str, PolarsDataType] = {
    "parameter": pl.String,
    "estimate": pl.Float64,
    "std_error": pl.Float64,
    "t_value": pl.Float64,
    "p_value": pl.Float64,
    "ci_lower": pl.Float64,
    "ci_upper": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
COEFFICIENT_COLUMNS = tuple(COEFFICIENT_SCHEMA)

DESIGN_TERM_SCHEMA: dict[str, PolarsDataType] = {
    "term": pl.String,
    "term_type": pl.String,
    "order": pl.Int64,
    "components_json": pl.String,
    "component_kinds_json": pl.String,
}
DESIGN_TERM_COLUMNS = tuple(DESIGN_TERM_SCHEMA)


@dataclass(frozen=True, slots=True)
class FactorialResult:
    """Structured result for one univariate factorial ANOVA/ANCOVA model."""

    status: ResultStatus
    reason: str | None
    design: FactorialDesign
    design_terms: pl.DataFrame
    effects: pl.DataFrame
    coefficients: pl.DataFrame
    diagnostics: pl.DataFrame
    observation_diagnostics: pl.DataFrame
    cells: pl.DataFrame
    exclusions: pl.DataFrame
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
        raise ValueError("robust_covariance must be one of none, hc0, hc1, hc2, hc3.")
    return normalized


def _design_term_table(design: FactorialDesign) -> pl.DataFrame:
    rows = [
        {
            "term": term.label,
            "term_type": term.term_type,
            "order": term.order,
            "components_json": json.dumps(term.columns, separators=(",", ":")),
            "component_kinds_json": json.dumps(term.kinds, separators=(",", ":")),
        }
        for term in design.terms
    ]
    return pl.DataFrame(rows, schema=DESIGN_TERM_SCHEMA) if rows else pl.DataFrame(schema=DESIGN_TERM_SCHEMA)


def _empty_result(
    *,
    status: ResultStatus,
    reason: str,
    design: FactorialDesign,
    design_terms: pl.DataFrame,
    cells: pl.DataFrame,
    exclusions: pl.DataFrame,
    model_summary: dict[str, Any],
    advisories: tuple[Advisory, ...],
    provenance: AnalysisProvenance,
) -> FactorialResult:
    from ruddy.factorial.diagnostics import (
        DIAGNOSTIC_SCHEMA,
        OBSERVATION_DIAGNOSTIC_SCHEMA_BASE,
    )

    id_dtype = exclusions.schema.get("observation_id", pl.String)
    obs_diag_schema = {**OBSERVATION_DIAGNOSTIC_SCHEMA_BASE, "observation_id": id_dtype}
    return FactorialResult(
        status=status,
        reason=reason,
        design=design,
        design_terms=design_terms,
        effects=pl.DataFrame(schema=EFFECT_SCHEMA),
        coefficients=pl.DataFrame(schema=COEFFICIENT_SCHEMA),
        diagnostics=pl.DataFrame(schema=DIAGNOSTIC_SCHEMA),
        observation_diagnostics=pl.DataFrame(schema=obs_diag_schema),
        cells=cells,
        exclusions=exclusions,
        model_summary=model_summary,
        advisories=advisories,
        provenance=provenance,
    )


def _safe_model_frame(
    model_frame: pl.DataFrame,
    design: FactorialDesign,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict[str, str]]:
    # pandas boundary: statsmodels/Patsy consume pandas; see docs/POLARS_MIGRATION_PLAN.md
    pandas_frame = model_frame.to_pandas()
    safe = pd.DataFrame(index=pandas_frame.index)
    safe["Y"] = pd.to_numeric(pandas_frame[design.response], errors="raise").astype(float)
    factor_map: dict[str, str] = {}
    covariate_map: dict[str, str] = {}
    safe_expression: dict[str, str] = {}
    for index, name in enumerate(design.factors):
        safe_name = f"F{index}"
        safe[safe_name] = pd.Categorical(pandas_frame[name])
        expression = f"C({safe_name}, Sum)"
        factor_map[safe_name] = name
        safe_expression[name] = expression
    for index, name in enumerate(design.covariates):
        safe_name = f"X{index}"
        safe[safe_name] = pd.to_numeric(pandas_frame[name], errors="raise").astype(float)
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
        raise ValueError("min_cell_n must be at least 1.")
    if max_factor_levels < 2:
        raise ValueError("max_factor_levels must be at least 2.")
    if max_design_cells < 1:
        raise ValueError("max_design_cells must be at least 1.")
    if max_design_columns < 2:
        raise ValueError("max_design_columns must be at least 2.")
    if not 0.0 < diagnostic_alpha < 1.0:
        raise ValueError("diagnostic_alpha must lie in (0, 1).")
    if condition_number_threshold <= 0.0:
        raise ValueError("condition_number_threshold must be greater than zero.")

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
    frame = dataset.frame.select(selected)
    complete = np.ones(frame.height, dtype=bool)
    numeric_columns = (design.response, *design.covariates)
    for name in numeric_columns:
        values = frame.get_column(name).cast(pl.Float64).fill_null(float("nan")).to_numpy()
        complete &= np.isfinite(values)
    for factor in design.factors:
        series = frame.get_column(factor)
        valid = series.is_not_null()
        if series.dtype.is_float():
            valid = valid & ~series.is_nan()
        complete &= valid.fill_null(False).to_numpy()

    source_rows = np.flatnonzero(complete).astype(np.int64)
    excluded_rows = np.flatnonzero(~complete).astype(np.int64)
    ids = dataset.observation_id_tuple
    exclusions = _exclusions_table(
        ids,
        excluded_rows,
        stage="factorial_complete_case",
        reason="missing_or_non_finite_model_value",
    )
    model_frame = frame.filter(pl.Series(complete))
    n = model_frame.height

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
        counts = len(dict.fromkeys(model_frame.get_column(factor).to_list()))
        if counts < 2:
            from ruddy.factorial.diagnostics import CELL_SCHEMA_BASE

            empty_cell_schema: dict[str, PolarsDataType] = {
                name: model_frame.schema[name] for name in design.factors
            }
            empty_cell_schema.update(CELL_SCHEMA_BASE)
            empty_cells = pl.DataFrame(schema=empty_cell_schema)
            return _empty_result(
                status=ResultStatus.DEGENERATE,
                reason="factor_has_fewer_than_two_levels",
                design=design,
                design_terms=design_terms,
                cells=empty_cells,
                exclusions=exclusions,
                model_summary={"n_complete_case": n, "factor": factor, "n_levels": counts},
                advisories=tuple(advisories),
                provenance=provenance,
            )
        if counts > max_factor_levels:
            raise ValueError(f"Factor {factor!r} has {counts} levels; maximum is {max_factor_levels}.")

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
    response_values = model_frame.get_column(design.response).cast(pl.Float64).to_numpy()
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
        raise ValueError(f"Factorial design has {n_design_columns} columns; maximum is {max_design_columns}.")
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
    ok_indices = [i for i, r in enumerate(effect_rows) if r["status"] == ResultStatus.OK.value]
    family_size = len(ok_indices)
    if family_size:
        p_values = np.array([effect_rows[i]["p_value"] for i in ok_indices], dtype=float)
        q_values = adjust_pvalues(p_values, correction)
        for idx, q in zip(ok_indices, q_values, strict=True):
            effect_rows[idx]["q_value"] = float(q) if q is not None and math.isfinite(q) else None
    for r in effect_rows:
        r["family_size"] = family_size
    effects = (
        pl.DataFrame(effect_rows, schema=EFFECT_SCHEMA) if effect_rows else pl.DataFrame(schema=EFFECT_SCHEMA)
    )

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
    coefficients = (
        pl.DataFrame(coefficient_rows, schema=COEFFICIENT_SCHEMA)
        if coefficient_rows
        else pl.DataFrame(schema=COEFFICIENT_SCHEMA)
    )

    diagnostics, observation_diagnostics = model_diagnostics(
        model,
        model_frame=model_frame,
        source_row_indices=source_rows,
        observation_ids=ids,
        factors=design.factors,
        alpha=diagnostic_alpha,
        condition_number_threshold=condition_number_threshold,
    )
    flagged = (
        diagnostics.filter(
            (pl.col("status") == ResultStatus.OK.value) & (pl.col("flagged") == True)  # noqa: E712
        )
        .get_column("diagnostic")
        .to_list()
    )
    if flagged:
        advisories.append(
            Advisory(
                code="factorial_diagnostics_flagged",
                message="One or more model diagnostics crossed their configured reference threshold.",
                context={"diagnostics": tuple(flagged), "alpha": diagnostic_alpha},
            )
        )
    if not observation_diagnostics.is_empty():
        n_influential = int(observation_diagnostics.get_column("is_influential").fill_null(False).sum())
        n_high_leverage = int(observation_diagnostics.get_column("is_high_leverage").fill_null(False).sum())
        n_large_residual = int(observation_diagnostics.get_column("is_large_residual").fill_null(False).sum())
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
    "COEFFICIENT_SCHEMA",
    "DESIGN_TERM_COLUMNS",
    "DESIGN_TERM_SCHEMA",
    "EFFECT_COLUMNS",
    "EFFECT_SCHEMA",
    "EXCLUSION_COLUMNS",
    "FactorialResult",
    "analyze_factorial",
]
