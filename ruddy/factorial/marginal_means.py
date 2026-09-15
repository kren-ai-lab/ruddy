"""Estimated marginal means and explicit pairwise contrasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrices
from scipy.stats import t as student_t
from statsmodels.api import OLS

from ruddy.core.enums import PAdjustMethod, ResultStatus
from ruddy.data import TabularDataset
from ruddy.factorial.design import FactorialDesign, build_factorial_design
from ruddy.results import AnalysisProvenance
from ruddy.statistics.multiple_testing import adjust_pvalues

MEAN_COLUMNS = (
    "response",
    "term",
    "levels_json",
    "estimate",
    "std_error",
    "df",
    "confidence_level",
    "ci_lower",
    "ci_upper",
    "weighting",
    "status",
    "reason",
)
CONTRAST_COLUMNS = (
    "response",
    "term",
    "levels_a_json",
    "levels_b_json",
    "estimate_difference",
    "std_error",
    "df",
    "t_value",
    "p_value",
    "q_value",
    "correction",
    "family_size",
    "confidence_level",
    "ci_lower",
    "ci_upper",
    "status",
    "reason",
)
EXCLUSION_COLUMNS = ("source_row_index", "observation_id", "stage", "reason")


@dataclass(frozen=True, slots=True)
class MarginalMeansResult:
    """Estimated marginal means plus pairwise contrasts for requested factor terms."""

    means: pd.DataFrame
    contrasts: pd.DataFrame
    exclusions: pd.DataFrame
    provenance: AnalysisProvenance


def _safe_model(
    dataset: TabularDataset,
    design: FactorialDesign,
) -> tuple[object, pd.DataFrame, np.ndarray, dict[str, str], dict[str, str], object]:
    selected = (design.response, *design.factors, *design.covariates)
    frame = dataset.select(selected)
    complete = np.ones(len(frame), dtype=bool)
    for name in (design.response, *design.covariates):
        complete &= np.isfinite(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
    for name in design.factors:
        complete &= frame[name].notna().to_numpy()
    model_frame = frame.loc[complete].reset_index(drop=True)
    safe = pd.DataFrame(index=model_frame.index)
    safe["Y"] = pd.to_numeric(model_frame[design.response], errors="raise").astype(float)
    factor_safe: dict[str, str] = {}
    covariate_safe: dict[str, str] = {}
    expression: dict[str, str] = {}
    for idx, name in enumerate(design.factors):
        safe_name = f"F{idx}"
        safe[safe_name] = pd.Categorical(model_frame[name])
        factor_safe[name] = safe_name
        expression[name] = f"C({safe_name}, Sum)"
    for idx, name in enumerate(design.covariates):
        safe_name = f"X{idx}"
        safe[safe_name] = pd.to_numeric(model_frame[name], errors="raise").astype(float)
        covariate_safe[name] = safe_name
        expression[name] = safe_name
    rhs = " + ".join(":".join(expression[name] for name in term.columns) for term in design.terms)
    formula = f"Y ~ {rhs}"
    response_matrix, design_matrix = dmatrices(formula, data=safe, return_type="dataframe")
    design_info = design_matrix.design_info
    model = OLS(response_matrix.iloc[:, 0], design_matrix).fit()
    return model, model_frame, complete, factor_safe, covariate_safe, design_info


def _normalize_terms(
    terms: tuple[str | tuple[str, ...], ...] | None,
    factors: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if terms is None:
        return tuple((factor,) for factor in factors)
    normalized: list[tuple[str, ...]] = []
    for term in terms:
        values = (term,) if isinstance(term, str) else tuple(str(value) for value in term)
        if not values or any(value not in factors for value in values):
            raise ValueError("Marginal-mean terms must contain declared factors only.")
        if len(set(values)) != len(values):
            raise ValueError("Marginal-mean terms cannot repeat factors.")
        if values not in normalized:
            normalized.append(values)
    return tuple(normalized)


def _grid_l_vectors(
    model,
    model_frame: pd.DataFrame,
    design: FactorialDesign,
    factor_safe: dict[str, str],
    covariate_safe: dict[str, str],
    design_info,
    term: tuple[str, ...],
) -> list[tuple[tuple[object, ...], np.ndarray]]:
    levels = {name: list(pd.unique(model_frame[name])) for name in design.factors}
    nuisance = tuple(name for name in design.factors if name not in term)
    target_combinations = list(product(*(levels[name] for name in term)))
    nuisance_combinations = list(product(*(levels[name] for name in nuisance))) if nuisance else [()]
    cov_means = {
        name: float(pd.to_numeric(model_frame[name], errors="raise").mean()) for name in design.covariates
    }
    result: list[tuple[tuple[object, ...], np.ndarray]] = []
    for target_values in target_combinations:
        rows: list[dict[str, object]] = []
        target_map = dict(zip(term, target_values, strict=True))
        for nuisance_values in nuisance_combinations:
            row: dict[str, object] = {}
            nuisance_map = dict(zip(nuisance, nuisance_values, strict=True))
            for factor in design.factors:
                row[factor_safe[factor]] = target_map.get(factor, nuisance_map.get(factor))
            for covariate in design.covariates:
                row[covariate_safe[covariate]] = cov_means[covariate]
            rows.append(row)
        grid = pd.DataFrame(rows)
        matrix = np.asarray(build_design_matrices([design_info], grid)[0], dtype=float)
        result.append((target_values, matrix.mean(axis=0)))
    return result


def analyze_marginal_means(
    dataset: TabularDataset,
    *,
    response: str | None = None,
    factors: tuple[str, ...] = (),
    covariates: tuple[str, ...] = (),
    interactions: tuple[tuple[str, ...], ...] = (),
    formula: str | None = None,
    terms: tuple[str | tuple[str, ...], ...] | None = None,
    confidence_level: float = 0.95,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    max_interaction_order: int = 3,
) -> MarginalMeansResult:
    """Estimate equal-weight marginal means and pairwise contrasts from an OLS model."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
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
    requested_terms = _normalize_terms(terms, design.factors)
    if not requested_terms:
        raise ValueError("Marginal means require at least one factor term.")
    model, model_frame, complete, factor_safe, covariate_safe, design_info = _safe_model(dataset, design)
    if model.df_resid <= 0:
        raise ValueError("Marginal means require positive residual degrees of freedom.")
    if np.linalg.matrix_rank(model.model.exog) < model.model.exog.shape[1]:
        raise ValueError("Marginal means require a full-rank fixed-effects design.")

    beta = np.asarray(model.params, dtype=float)
    cov_beta = np.asarray(model.cov_params(), dtype=float)
    df = float(model.df_resid)
    alpha = 1.0 - confidence_level
    tcrit = float(student_t.ppf(1.0 - alpha / 2.0, df))
    mean_rows: list[dict] = []
    contrast_rows: list[dict] = []

    for term in requested_terms:
        term_label = ":".join(term)
        l_vectors = _grid_l_vectors(
            model, model_frame, design, factor_safe, covariate_safe, design_info, term
        )
        for levels, lvec in l_vectors:
            estimate = float(lvec @ beta)
            variance = float(lvec @ cov_beta @ lvec)
            se = float(np.sqrt(max(variance, 0.0)))
            half = tcrit * se
            mean_rows.append(
                {
                    "response": design.response,
                    "term": term_label,
                    "levels_json": json.dumps(
                        dict(zip(term, levels, strict=True)), default=str, separators=(",", ":")
                    ),
                    "estimate": estimate,
                    "std_error": se,
                    "df": df,
                    "confidence_level": confidence_level,
                    "ci_lower": estimate - half,
                    "ci_upper": estimate + half,
                    "weighting": "equal",
                    "status": ResultStatus.OK.value,
                    "reason": None,
                }
            )
        family_rows: list[dict] = []
        for (levels_a, la), (levels_b, lb) in combinations(l_vectors, 2):
            contrast = lb - la
            estimate = float(contrast @ beta)
            variance = float(contrast @ cov_beta @ contrast)
            se = float(np.sqrt(max(variance, 0.0)))
            t_value = estimate / se if se > 0 else (np.inf if estimate != 0 else 0.0)
            p_value = float(2.0 * student_t.sf(abs(t_value), df))
            half = tcrit * se
            family_rows.append(
                {
                    "response": design.response,
                    "term": term_label,
                    "levels_a_json": json.dumps(
                        dict(zip(term, levels_a, strict=True)), default=str, separators=(",", ":")
                    ),
                    "levels_b_json": json.dumps(
                        dict(zip(term, levels_b, strict=True)), default=str, separators=(",", ":")
                    ),
                    "estimate_difference": estimate,
                    "std_error": se,
                    "df": df,
                    "t_value": float(t_value),
                    "p_value": p_value,
                    "q_value": np.nan,
                    "correction": correction.value,
                    "family_size": len(l_vectors) * (len(l_vectors) - 1) // 2,
                    "confidence_level": confidence_level,
                    "ci_lower": estimate - half,
                    "ci_upper": estimate + half,
                    "status": ResultStatus.OK.value,
                    "reason": None,
                }
            )
        if family_rows:
            q_values = adjust_pvalues([row["p_value"] for row in family_rows], method=correction)
            for row, q_value in zip(family_rows, q_values, strict=True):
                row["q_value"] = float(q_value) if q_value is not None else np.nan
            contrast_rows.extend(family_rows)

    exclusions_idx = np.flatnonzero(~complete)
    exclusions = pd.DataFrame(
        {
            "source_row_index": exclusions_idx.astype(np.int64),
            "observation_id": dataset.observation_ids.take(exclusions_idx).to_list(),
            "stage": "marginal_means_complete_case",
            "reason": "missing_or_non_finite_model_value",
        },
        columns=EXCLUSION_COLUMNS,
    )
    provenance = AnalysisProvenance(
        analysis="marginal_means",
        parameters={
            "response": design.response,
            "factors": design.factors,
            "covariates": design.covariates,
            "interactions": design.interactions,
            "requested_formula": design.requested_formula,
            "resolved_formula": design.resolved_formula,
            "terms": requested_terms,
            "weighting": "equal",
            "confidence_level": confidence_level,
            "p_adjust": correction.value,
        },
        input_summary={
            "n_observations": dataset.n_observations,
            "n_complete_case": int(complete.sum()),
            "n_excluded": int((~complete).sum()),
        },
    )
    return MarginalMeansResult(
        means=pd.DataFrame(mean_rows, columns=MEAN_COLUMNS),
        contrasts=pd.DataFrame(contrast_rows, columns=CONTRAST_COLUMNS),
        exclusions=exclusions,
        provenance=provenance,
    )


__all__ = ["MarginalMeansResult", "analyze_marginal_means"]
