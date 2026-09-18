"""Low/moderate-dimensional multivariate analysis of variance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from patsy import dmatrices  # pyrefly: ignore[missing-module-attribute]
from polars._typing import PolarsDataType
from statsmodels.multivariate.manova import MANOVA

from ruddy.core.enums import ColumnKind, ColumnRole, ResultStatus
from ruddy.data import TabularDataset
from ruddy.projections.preprocessing import _exclusions_table
from ruddy.results import AnalysisProvenance

MANOVA_TEST_SCHEMA: dict[str, PolarsDataType] = {
    "term": pl.String,
    "statistic": pl.String,
    "value": pl.Float64,
    "num_df": pl.Float64,
    "den_df": pl.Float64,
    "f_value": pl.Float64,
    "p_value": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
MANOVA_TEST_COLUMNS = tuple(MANOVA_TEST_SCHEMA)

FACTOR_LEVEL_SCHEMA: dict[str, PolarsDataType] = {
    "factor": pl.String,
    "level": pl.String,
    "n": pl.Int64,
    "status": pl.String,
    "reason": pl.String,
}
FACTOR_LEVEL_COLUMNS = tuple(FACTOR_LEVEL_SCHEMA)


@dataclass(frozen=True, slots=True)
class MANOVAResult:
    """Structured MANOVA statistics with complete-case accounting."""

    status: ResultStatus
    reason: str | None
    tests: pl.DataFrame
    factor_levels: pl.DataFrame
    exclusions: pl.DataFrame
    model_summary: dict[str, Any]
    provenance: AnalysisProvenance


def _normalize_columns(values: Iterable[str], label: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} cannot contain duplicates.")
    return normalized


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _empty_tests() -> pl.DataFrame:
    return pl.DataFrame(schema=MANOVA_TEST_SCHEMA)


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
    frame = dataset.frame.select(selected)
    complete = np.ones(frame.height, dtype=bool)
    for name in (*response_names, *covariate_names):
        values = frame.get_column(name).cast(pl.Float64).fill_null(float("nan")).to_numpy()
        complete &= np.isfinite(values)
    for name in factor_names:
        col = frame.get_column(name)
        mask = col.is_not_null()
        if col.dtype.is_float():
            mask &= ~col.is_nan()
        complete &= mask.to_numpy()
    model_frame = frame.filter(pl.Series(complete))
    excluded_rows = np.flatnonzero(~complete).astype(np.int64)
    exclusions = _exclusions_table(
        dataset.observation_id_tuple,
        excluded_rows,
        stage="manova_complete_case",
        reason="missing_or_non_finite_model_value",
    )
    n = model_frame.height

    level_rows: list[dict[str, Any]] = []
    for factor in factor_names:
        unique_levels = list(dict.fromkeys(model_frame.get_column(factor).to_list()))
        if len(unique_levels) < 2:
            return MANOVAResult(
                status=ResultStatus.DEGENERATE,
                reason="factor_has_fewer_than_two_levels",
                tests=_empty_tests(),
                factor_levels=pl.DataFrame(level_rows, schema=FACTOR_LEVEL_SCHEMA),
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
        if len(unique_levels) > max_factor_levels:
            raise ValueError(
                f"MANOVA factor {factor!r} has {len(unique_levels)} levels; maximum is {max_factor_levels}."
            )
        vc = model_frame.get_column(factor).value_counts()
        counts_dict = {row[factor]: int(row["count"]) for row in vc.iter_rows(named=True)}
        for level in unique_levels:
            count = counts_dict[level]
            level_rows.append(
                {
                    "factor": factor,
                    "level": str(level),
                    "n": int(count),
                    "status": ResultStatus.OK.value
                    if int(count) >= min_level_n
                    else ResultStatus.DEGENERATE.value,
                    "reason": None if int(count) >= min_level_n else "level_too_small",
                }
            )
    factor_levels = pl.DataFrame(level_rows, schema=FACTOR_LEVEL_SCHEMA)
    if not factor_levels.is_empty() and (factor_levels.get_column("status") != ResultStatus.OK.value).any():
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

    # pandas boundary: statsmodels/Patsy consume pandas; see docs/POLARS_MIGRATION_PLAN.md
    pandas_frame = model_frame.to_pandas()
    safe = pd.DataFrame(index=pandas_frame.index)
    response_map: dict[str, str] = {}
    factor_map: dict[str, str] = {}
    covariate_map: dict[str, str] = {}
    for index, name in enumerate(response_names):
        safe_name = f"Y{index}"
        safe[safe_name] = pd.to_numeric(pandas_frame[name], errors="raise").astype(float)
        response_map[safe_name] = name
    for index, name in enumerate(factor_names):
        safe_name = f"F{index}"
        safe[safe_name] = pandas_frame[name].astype("category")
        factor_map[f"C({safe_name})"] = name
    for index, name in enumerate(covariate_names):
        safe_name = f"X{index}"
        safe[safe_name] = pd.to_numeric(pandas_frame[name], errors="raise").astype(float)
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
                    "value": _finite_or_none(row["Value"]),
                    "num_df": _finite_or_none(row["Num DF"]),
                    "den_df": _finite_or_none(row["Den DF"]),
                    "f_value": _finite_or_none(row["F Value"]),
                    "p_value": _finite_or_none(row["Pr > F"]),
                    "status": ResultStatus.OK.value,
                    "reason": None,
                }
            )
    tests = pl.DataFrame(rows, schema=MANOVA_TEST_SCHEMA)
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
