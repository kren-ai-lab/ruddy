"""Cell-level diagnostics for categorical contingency tables."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from ruddy.bivariate.associations import _contingency, _eligible_categorical
from ruddy.core.enums import PAdjustMethod
from ruddy.results import AnalysisProvenance
from ruddy.statistics import apply_multiple_testing, bias_corrected_cramers_v, expected_count_diagnostics

if TYPE_CHECKING:
    from ruddy.data import TabularDataset

SUMMARY_COLUMNS = (
    "column_x",
    "column_y",
    "n_total",
    "n_used",
    "n_x_levels",
    "n_y_levels",
    "chi_square",
    "df",
    "p_value",
    "q_value",
    "family_id",
    "family_size",
    "correction",
    "cramers_v",
    "min_expected_count",
    "n_expected_lt5",
    "fraction_expected_lt5",
    "status",
    "reason",
)
CELL_COLUMNS = (
    "column_x",
    "column_y",
    "level_x",
    "level_y",
    "observed",
    "expected",
    "pearson_residual",
    "standardized_residual",
    "chi_square_contribution",
    "chi_square_contribution_fraction",
    "row_fraction",
    "column_fraction",
    "status",
    "reason",
)


@dataclass(frozen=True, slots=True)
class ContingencyDiagnosticsResult:
    summary: pd.DataFrame
    cells: pd.DataFrame
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
        msg = "max_category_levels must be at least 2."
        raise ValueError(msg)
    candidates = _eligible_categorical(dataset)
    selected_pairs = (
        tuple(combinations(candidates, 2)) if pairs is None else tuple((str(x), str(y)) for x, y in pairs)
    )
    invalid = [(x, y) for x, y in selected_pairs if x == y or x not in candidates or y not in candidates]
    if invalid:
        msg = f"Invalid contingency pairs: {invalid}."
        raise ValueError(msg)
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    frame = dataset.to_frame()
    summaries: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for x, y in selected_pairs:
        x_levels, y_levels, counts, n_used = _contingency(frame, x, y)
        base: dict[str, Any] = {
            "column_x": x,
            "column_y": y,
            "n_total": dataset.n_observations,
            "n_used": n_used,
            "n_x_levels": len(x_levels),
            "n_y_levels": len(y_levels),
            "chi_square": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "q_value": np.nan,
            "family_id": "contingency:chi_square",
            "family_size": 0,
            "correction": correction.value,
            "cramers_v": np.nan,
            "min_expected_count": np.nan,
            "n_expected_lt5": np.nan,
            "fraction_expected_lt5": np.nan,
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
            df=float(df),
            p_value=float(p_value),
            cramers_v=np.nan if effect is None else effect,
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
                    pearson = standardized = contribution = fraction = np.nan
                    status, reason = "degenerate", "zero_expected_count"
                else:
                    pearson = (observed - exp) / np.sqrt(exp)
                    adjustment = (1.0 - row_props[i]) * (1.0 - column_props[j])
                    standardized = pearson / np.sqrt(adjustment) if adjustment > 0.0 else np.nan
                    contribution = (observed - exp) ** 2 / exp
                    fraction = contribution / chi2 if chi2 > 0.0 else 0.0
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
                        "row_fraction": observed / row_totals[i] if row_totals[i] else np.nan,
                        "column_fraction": observed / column_totals[j] if column_totals[j] else np.nan,
                        "status": status,
                        "reason": reason,
                    }
                )
    summary = pd.DataFrame(summaries, columns=pd.Index(SUMMARY_COLUMNS))
    if not summary.empty:
        summary = apply_multiple_testing(summary, correction)
    cell_table = pd.DataFrame(cells, columns=pd.Index(CELL_COLUMNS))
    provenance = AnalysisProvenance(
        analysis="contingency_diagnostics",
        parameters={
            "max_category_levels": max_category_levels,
            "p_adjust": correction.value,
            "pearson_residual": "(observed-expected)/sqrt(expected)",
            "standardized_residual": "haberman_adjusted",
            "chi_square_contribution": "(observed-expected)^2/expected",
        },
        input_summary={"n_observations": dataset.n_observations, "n_columns": dataset.n_columns},
    )
    return ContingencyDiagnosticsResult(summary=summary, cells=cell_table, provenance=provenance)
