"""Cell-level diagnostics for categorical contingency tables."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from polars._typing import PolarsDataType
from scipy import stats

from ruddy.bivariate.associations import _contingency, _eligible_categorical
from ruddy.core.enums import PAdjustMethod
from ruddy.data import TabularDataset
from ruddy.results import AnalysisProvenance
from ruddy.statistics import apply_multiple_testing, bias_corrected_cramers_v, expected_count_diagnostics

SUMMARY_SCHEMA: dict[str, PolarsDataType] = {
    "column_x": pl.String,
    "column_y": pl.String,
    "n_total": pl.Int64,
    "n_used": pl.Int64,
    "n_x_levels": pl.Int64,
    "n_y_levels": pl.Int64,
    "chi_square": pl.Float64,
    "df": pl.Int64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "correction": pl.String,
    "cramers_v": pl.Float64,
    "min_expected_count": pl.Float64,
    "n_expected_lt5": pl.Int64,
    "fraction_expected_lt5": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
SUMMARY_COLUMNS: tuple[str, ...] = tuple(SUMMARY_SCHEMA)

CELL_SCHEMA: dict[str, PolarsDataType] = {
    "column_x": pl.String,
    "column_y": pl.String,
    "level_x": pl.String,
    "level_y": pl.String,
    "observed": pl.Float64,
    "expected": pl.Float64,
    "pearson_residual": pl.Float64,
    "standardized_residual": pl.Float64,
    "chi_square_contribution": pl.Float64,
    "chi_square_contribution_fraction": pl.Float64,
    "row_fraction": pl.Float64,
    "column_fraction": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
CELL_COLUMNS: tuple[str, ...] = tuple(CELL_SCHEMA)


@dataclass(frozen=True, slots=True)
class ContingencyDiagnosticsResult:
    summary: pl.DataFrame
    cells: pl.DataFrame
    provenance: AnalysisProvenance


def analyze_contingency_diagnostics(
    dataset: TabularDataset,
    *,
    pairs: tuple[tuple[str, str], ...] | None = None,
    max_category_levels: int = 50,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> ContingencyDiagnosticsResult:
    """Return expected counts and cell residuals for categorical associations."""
    if max_category_levels < 2:
        raise ValueError("max_category_levels must be at least 2.")
    candidates = _eligible_categorical(dataset)
    selected_pairs = (
        tuple(combinations(candidates, 2)) if pairs is None else tuple((str(x), str(y)) for x, y in pairs)
    )
    invalid = [(x, y) for x, y in selected_pairs if x == y or x not in candidates or y not in candidates]
    if invalid:
        raise ValueError(f"Invalid contingency pairs: {invalid}.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    frame = dataset.frame
    summaries: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for x, y in selected_pairs:
        x_levels, y_levels, counts, n_used = _contingency(frame, x, y)
        base: dict[str, Any] = {
            "column_x": x,
            "column_y": y,
            "n_total": dataset.frame.height,
            "n_used": n_used,
            "n_x_levels": len(x_levels),
            "n_y_levels": len(y_levels),
            "chi_square": None,
            "df": None,
            "p_value": None,
            "q_value": None,
            "family_id": "contingency:chi_square",
            "family_size": 0,
            "correction": correction.value,
            "cramers_v": None,
            "min_expected_count": None,
            "n_expected_lt5": None,
            "fraction_expected_lt5": None,
            "status": "ok",
            "reason": None,
        }
        if len(x_levels) > max_category_levels or len(y_levels) > max_category_levels:
            base["status"] = "skipped"
            base["reason"] = "feature_levels_exceed_max_category_levels"
            summaries.append(base)
            continue
        if counts.shape[0] < 2 or counts.shape[1] < 2 or n_used <= 0:
            base["status"] = "degenerate"
            base["reason"] = "insufficient_contingency_dimensions"
            summaries.append(base)
            continue
        chi2, p_value, df, expected = stats.chi2_contingency(counts, correction=False)
        diagnostics = expected_count_diagnostics(expected)
        effect = bias_corrected_cramers_v(float(chi2), n_used, counts.shape[0], counts.shape[1])
        base.update(
            chi_square=float(chi2),
            df=int(df),
            p_value=float(p_value),
            cramers_v=effect,
            min_expected_count=diagnostics["min_expected_count"],
            n_expected_lt5=diagnostics["n_expected_lt5"],
            fraction_expected_lt5=diagnostics["fraction_expected_lt5"],
        )
        summaries.append(base)
        row_totals = counts.sum(axis=1).astype(float)
        column_totals = counts.sum(axis=0).astype(float)
        row_props = row_totals / n_used
        column_props = column_totals / n_used
        for i, x_level in enumerate(x_levels):
            for j, y_level in enumerate(y_levels):
                observed = float(counts[i, j])
                exp = float(expected[i, j])
                if exp <= 0.0:
                    pearson = standardized = contribution = fraction = None
                    status, reason = "degenerate", "zero_expected_count"
                else:
                    pearson = float((observed - exp) / np.sqrt(exp))
                    adjustment = (1.0 - row_props[i]) * (1.0 - column_props[j])
                    standardized = float(pearson / np.sqrt(adjustment)) if adjustment > 0.0 else None
                    contribution = float((observed - exp) ** 2 / exp)
                    fraction = float(contribution / chi2) if chi2 > 0.0 else 0.0
                    status, reason = "ok", None
                cells.append(
                    {
                        "column_x": x,
                        "column_y": y,
                        "level_x": x_level,
                        "level_y": y_level,
                        "observed": observed,
                        "expected": exp,
                        "pearson_residual": pearson,
                        "standardized_residual": standardized,
                        "chi_square_contribution": contribution,
                        "chi_square_contribution_fraction": fraction,
                        "row_fraction": float(observed / row_totals[i]) if row_totals[i] else None,
                        "column_fraction": float(observed / column_totals[j]) if column_totals[j] else None,
                        "status": status,
                        "reason": reason,
                    }
                )
    summary_table = pl.DataFrame(summaries, schema=SUMMARY_SCHEMA)
    summary = (
        summary_table if summary_table.height == 0 else apply_multiple_testing(summary_table, correction)
    )
    cell_table = pl.DataFrame(cells, schema=CELL_SCHEMA)
    provenance = AnalysisProvenance(
        analysis="contingency_diagnostics",
        parameters={
            "max_category_levels": max_category_levels,
            "p_adjust": correction.value,
            "pearson_residual": "(observed-expected)/sqrt(expected)",
            "standardized_residual": "haberman_adjusted",
            "chi_square_contribution": "(observed-expected)^2/expected",
        },
        input_summary={"n_observations": dataset.frame.height, "n_columns": dataset.frame.width},
    )
    return ContingencyDiagnosticsResult(summary=summary, cells=cell_table, provenance=provenance)
