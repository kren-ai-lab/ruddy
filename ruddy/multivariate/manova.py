"""Low/moderate-dimensional multivariate analysis of variance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import dmatrices  # ty: ignore[unresolved-import]
from statsmodels.multivariate.manova import MANOVA

from ruddy.core.enums import ColumnKind, ColumnRole, ResultStatus
from ruddy.data import TabularDataset
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class MANOVAResult:
    """Structured MANOVA statistics with complete-case accounting."""

    status: ResultStatus
    reason: str | None
    tests: pd.DataFrame
    factor_levels: pd.DataFrame
    exclusions: pd.DataFrame
    model_summary: dict[str, Any]
    provenance: AnalysisProvenance


def _normalize_columns(values: Iterable[str], label: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} cannot contain duplicates.")
    return normalized


def _empty_tests() -> pd.DataFrame:
    return pd.DataFrame(
        columns=pd.Index(
            (
                "term",
                "statistic",
                "value",
                "num_df",
                "den_df",
                "f_value",
                "p_value",
                "status",
                "reason",
            )
        )
    )


def analyze_manova(
    dataset: TabularDataset,
    *,
    responses: Iterable[str],
    factors: Iterable[str],
    covariates: Iterable[str] = (),
    max_responses: int = 20,
    max_factor_levels: int = 20,
    min_level_n: int = 3,
) -> MANOVAResult:
    """Fit a main-effects MANOVA on explicitly selected responses/factors/covariates.

    Interactions are intentionally deferred to the factorial engine. Numeric factors
    are treated categorically when their statistical role is ``factor``.
    """
    if max_responses < 2:
        raise ValueError("max_responses must be at least 2.")
    if max_factor_levels < 2:
        raise ValueError("max_factor_levels must be at least 2.")
    if min_level_n < 2:
        raise ValueError("min_level_n must be at least 2.")

    response_names = _normalize_columns(responses, "responses")
    factor_names = _normalize_columns(factors, "factors")
    covariate_names = _normalize_columns(covariates, "covariates")
    if len(response_names) < 2:
        raise ValueError("MANOVA requires at least two numeric responses.")
    if len(response_names) > max_responses:
        raise ValueError(
            f"MANOVA is limited to {max_responses} responses per run; use explicit "
            "dimensionality reduction before fitting a higher-dimensional response space."
        )
    if not factor_names:
        raise ValueError("MANOVA requires at least one factor.")
    overlap = set(response_names) & (set(factor_names) | set(covariate_names))
    overlap |= set(factor_names) & set(covariate_names)
    if overlap:
        raise ValueError(f"MANOVA roles overlap for columns: {sorted(overlap)}.")

    unknown = [
        name for name in (*response_names, *factor_names, *covariate_names) if name not in dataset.columns
    ]
    if unknown:
        raise ValueError(f"Unknown MANOVA columns: {unknown}.")
    for response in response_names:
        if dataset.kind_of(response) is not ColumnKind.NUMERIC:
            raise ValueError(f"MANOVA response {response!r} must be numeric.")
    for covariate in covariate_names:
        if dataset.kind_of(covariate) is not ColumnKind.NUMERIC:
            raise ValueError(f"MANOVA covariate {covariate!r} must be numeric.")
    for factor in factor_names:
        kind = dataset.kind_of(factor)
        role = dataset.role_of(factor)
        if kind not in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN} and role is not ColumnRole.FACTOR:
            raise ValueError(
                f"MANOVA factor {factor!r} must be categorical/boolean or explicitly assigned the factor role."
            )

    selected = (*response_names, *factor_names, *covariate_names)
    frame = dataset.select(selected)
    finite_numeric = np.ones(len(frame), dtype=bool)
    for name in (*response_names, *covariate_names):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        finite_numeric &= np.isfinite(values)
    factor_present = np.ones(len(frame), dtype=bool)
    for name in factor_names:
        factor_present &= frame[name].notna().to_numpy()
    complete = finite_numeric & factor_present
    source_rows = np.flatnonzero(complete)
    excluded_rows = np.flatnonzero(~complete)
    ids = dataset.observation_ids
    exclusions = pd.DataFrame(
        {
            "source_row_index": excluded_rows.astype(np.int64),
            "observation_id": ids.take(excluded_rows).to_list(),
            "stage": "manova_complete_case",
            "reason": "missing_or_non_finite_model_value",
        }
    )
    model_frame = frame.loc[complete].reset_index(drop=True)
    n = len(model_frame)

    level_rows: list[dict[str, Any]] = []
    for factor in factor_names:
        counts = model_frame[factor].value_counts(dropna=False, sort=False)
        if len(counts) < 2:
            return MANOVAResult(
                status=ResultStatus.DEGENERATE,
                reason="factor_has_fewer_than_two_levels",
                tests=_empty_tests(),
                factor_levels=pd.DataFrame(level_rows),
                exclusions=exclusions,
                model_summary={"factor": factor, "n_complete_case": n},
                provenance=AnalysisProvenance(
                    analysis="manova",
                    parameters={
                        "responses": response_names,
                        "factors": factor_names,
                        "covariates": covariate_names,
                    },
                    input_summary={"n_observations": dataset.n_observations, "n_complete_case": n},
                ),
            )
        if len(counts) > max_factor_levels:
            raise ValueError(
                f"MANOVA factor {factor!r} has {len(counts)} levels; maximum is {max_factor_levels}."
            )
        for level, count in counts.items():
            level_rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "n": int(count),
                    "status": ResultStatus.OK.value
                    if int(count) >= min_level_n
                    else ResultStatus.DEGENERATE.value,
                    "reason": None if int(count) >= min_level_n else "level_too_small",
                }
            )
    factor_levels = pd.DataFrame(level_rows)
    if not factor_levels.empty and (factor_levels["status"] != ResultStatus.OK.value).any():
        reason = "factor_level_too_small"
        provenance = AnalysisProvenance(
            analysis="manova",
            parameters={
                "responses": response_names,
                "factors": factor_names,
                "covariates": covariate_names,
                "max_responses": int(max_responses),
                "max_factor_levels": int(max_factor_levels),
                "min_level_n": int(min_level_n),
                "interactions": False,
            },
            input_summary={
                "n_observations": dataset.n_observations,
                "n_complete_case": n,
                "n_excluded": len(excluded_rows),
            },
        )
        return MANOVAResult(
            status=ResultStatus.DEGENERATE,
            reason=reason,
            tests=_empty_tests(),
            factor_levels=factor_levels,
            exclusions=exclusions,
            model_summary={"n_complete_case": n},
            provenance=provenance,
        )

    safe = pd.DataFrame(index=model_frame.index)
    response_map: dict[str, str] = {}
    factor_map: dict[str, str] = {}
    covariate_map: dict[str, str] = {}
    for index, name in enumerate(response_names):
        safe_name = f"Y{index}"
        safe[safe_name] = pd.to_numeric(model_frame[name], errors="raise").astype(float)
        response_map[safe_name] = name
    for index, name in enumerate(factor_names):
        safe_name = f"F{index}"
        safe[safe_name] = model_frame[name].astype("category")
        factor_map[f"C({safe_name})"] = name
    for index, name in enumerate(covariate_names):
        safe_name = f"X{index}"
        safe[safe_name] = pd.to_numeric(model_frame[name], errors="raise").astype(float)
        covariate_map[safe_name] = name

    response_matrix = safe[list(response_map)].to_numpy(dtype=float)
    response_rank = int(np.linalg.matrix_rank(response_matrix - response_matrix.mean(axis=0)))
    if response_rank < len(response_names):
        provenance = AnalysisProvenance(
            analysis="manova",
            parameters={"responses": response_names, "factors": factor_names, "covariates": covariate_names},
            input_summary={
                "n_observations": dataset.n_observations,
                "n_complete_case": n,
                "response_rank": response_rank,
            },
        )
        return MANOVAResult(
            status=ResultStatus.DEGENERATE,
            reason="rank_deficient_response_matrix",
            tests=_empty_tests(),
            factor_levels=factor_levels,
            exclusions=exclusions,
            model_summary={"n_complete_case": n, "response_rank": response_rank},
            provenance=provenance,
        )

    lhs = " + ".join(response_map)
    rhs_terms = [f"C(F{index})" for index in range(len(factor_names))]
    rhs_terms.extend(f"X{index}" for index in range(len(covariate_names)))
    formula = f"{lhs} ~ " + " + ".join(rhs_terms)
    _endog_design, exog_design = dmatrices(formula, safe, return_type="dataframe")
    exog = np.asarray(exog_design, dtype=float)
    exog_rank = int(np.linalg.matrix_rank(exog))
    if exog_rank < exog.shape[1]:
        provenance = AnalysisProvenance(
            analysis="manova",
            parameters={"responses": response_names, "factors": factor_names, "covariates": covariate_names},
            input_summary={
                "n_observations": dataset.n_observations,
                "n_complete_case": n,
                "exog_rank": exog_rank,
            },
        )
        return MANOVAResult(
            status=ResultStatus.DEGENERATE,
            reason="rank_deficient_design",
            tests=_empty_tests(),
            factor_levels=factor_levels,
            exclusions=exclusions,
            model_summary={
                "n_complete_case": n,
                "exog_rank": exog_rank,
                "n_design_columns": int(exog.shape[1]),
            },
            provenance=provenance,
        )
    if n <= exog_rank + len(response_names):
        provenance = AnalysisProvenance(
            analysis="manova",
            parameters={"responses": response_names, "factors": factor_names, "covariates": covariate_names},
            input_summary={
                "n_observations": dataset.n_observations,
                "n_complete_case": n,
                "exog_rank": exog_rank,
            },
        )
        return MANOVAResult(
            status=ResultStatus.SKIPPED,
            reason="insufficient_residual_degrees_of_freedom",
            tests=_empty_tests(),
            factor_levels=factor_levels,
            exclusions=exclusions,
            model_summary={"n_complete_case": n, "exog_rank": exog_rank},
            provenance=provenance,
        )

    try:
        model = MANOVA.from_formula(formula, data=safe)
        mv = model.mv_test()
    except (ValueError, np.linalg.LinAlgError):
        provenance = AnalysisProvenance(
            analysis="manova",
            parameters={"responses": response_names, "factors": factor_names, "covariates": covariate_names},
            input_summary={
                "n_observations": dataset.n_observations,
                "n_complete_case": n,
                "exog_rank": exog_rank,
            },
        )
        return MANOVAResult(
            status=ResultStatus.DEGENERATE,
            reason="manova_fit_failed",
            tests=_empty_tests(),
            factor_levels=factor_levels,
            exclusions=exclusions,
            model_summary={"n_complete_case": n, "exog_rank": exog_rank},
            provenance=provenance,
        )
    rows: list[dict[str, Any]] = []
    term_map = {"Intercept": "Intercept", **factor_map, **covariate_map}
    for safe_term, result in mv.results.items():
        term = term_map.get(safe_term, safe_term)
        stat_table = result["stat"]
        for statistic_name, row in stat_table.iterrows():
            rows.append(
                {
                    "term": term,
                    "statistic": str(statistic_name),
                    "value": float(row["Value"]),
                    "num_df": float(row["Num DF"]),
                    "den_df": float(row["Den DF"]),
                    "f_value": float(row["F Value"]),
                    "p_value": float(row["Pr > F"]),
                    "status": ResultStatus.OK.value,
                    "reason": None,
                }
            )
    tests = pd.DataFrame(rows)
    model_summary = {
        "formula_basis": "main_effects_only",
        "n_complete_case": int(n),
        "n_excluded": len(excluded_rows),
        "n_responses": len(response_names),
        "response_rank": response_rank,
        "n_design_columns": int(exog.shape[1]),
        "design_rank": exog_rank,
        "responses": response_names,
        "factors": factor_names,
        "covariates": covariate_names,
    }
    provenance = AnalysisProvenance(
        analysis="manova",
        parameters={
            "responses": response_names,
            "factors": factor_names,
            "covariates": covariate_names,
            "max_responses": int(max_responses),
            "max_factor_levels": int(max_factor_levels),
            "min_level_n": int(min_level_n),
            "interactions": False,
            "test_statistics": (
                "Wilks' lambda",
                "Pillai's trace",
                "Hotelling-Lawley trace",
                "Roy's greatest root",
            ),
        },
        input_summary={
            "n_observations": dataset.n_observations,
            "n_complete_case": int(n),
            "n_excluded": len(excluded_rows),
            "n_responses": len(response_names),
            "design_rank": exog_rank,
            "response_rank": response_rank,
        },
    )
    return MANOVAResult(
        status=ResultStatus.OK,
        reason=None,
        tests=tests,
        factor_levels=factor_levels,
        exclusions=exclusions,
        model_summary=model_summary,
        provenance=provenance,
    )
