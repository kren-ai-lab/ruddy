"""Explicit post-hoc pairwise comparisons for grouped numeric responses."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import studentized_range

from ruddy.core.enums import ColumnKind, ColumnRole, ResultStatus
from ruddy.data import TabularDataset
from ruddy.results import AnalysisProvenance

PAIRWISE_COLUMNS = (
    "response",
    "factor",
    "method",
    "group_a",
    "group_b",
    "n_a",
    "n_b",
    "mean_a",
    "mean_b",
    "mean_difference",
    "std_error",
    "df",
    "statistic",
    "p_value",
    "confidence_level",
    "ci_lower",
    "ci_upper",
    "correction",
    "family_size",
    "status",
    "reason",
)


@dataclass(frozen=True, slots=True)
class PosthocResult:
    """Structured Tukey HSD / Games-Howell comparisons."""

    comparisons: pd.DataFrame
    provenance: AnalysisProvenance


def _validate_response_factor(dataset: TabularDataset, response: str, factor: str) -> None:
    if response not in dataset.columns:
        raise ValueError(f"Unknown response column: {response!r}.")
    if factor not in dataset.columns:
        raise ValueError(f"Unknown factor column: {factor!r}.")
    if dataset.kind_of(response) is not ColumnKind.NUMERIC:
        raise ValueError("Post-hoc comparisons require a numeric response.")
    if dataset.role_of(response) in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED}:
        raise ValueError("Post-hoc response cannot be identifier/excluded.")
    if dataset.role_of(factor) is not ColumnRole.FACTOR and dataset.kind_of(factor) not in {
        ColumnKind.CATEGORICAL,
        ColumnKind.BOOLEAN,
    }:
        raise ValueError("Post-hoc factor must be categorical/boolean or explicitly declared as factor.")


def _group_values(dataset: TabularDataset, response: str, factor: str) -> dict[object, np.ndarray]:
    frame = dataset.select([response, factor])
    numeric = pd.to_numeric(frame[response], errors="coerce").to_numpy(dtype=float)
    labels = frame[factor].to_numpy()
    valid = np.isfinite(numeric) & pd.notna(labels)
    result: dict[object, np.ndarray] = {}
    for level in pd.unique(labels[valid]):
        values = numeric[valid & (labels == level)]
        result[level] = np.asarray(values, dtype=float)
    return result


def _tukey_rows(
    groups: dict[object, np.ndarray], *, response: str, factor: str, confidence_level: float
) -> list[dict]:
    levels = list(groups)
    k = len(levels)
    n_total = sum(len(groups[level]) for level in levels)
    df_error = n_total - k
    if k < 2 or df_error <= 0:
        return []
    sse = sum(float(np.sum((groups[level] - groups[level].mean()) ** 2)) for level in levels)
    mse = sse / df_error if df_error > 0 else np.nan
    alpha = 1.0 - confidence_level
    qcrit = float(studentized_range.ppf(1.0 - alpha, k, df_error)) if np.isfinite(mse) else np.nan
    rows: list[dict] = []
    family_size = k * (k - 1) // 2
    for a, b in combinations(levels, 2):
        xa, xb = groups[a], groups[b]
        mean_a, mean_b = float(xa.mean()), float(xb.mean())
        diff = mean_b - mean_a
        if len(xa) < 2 or len(xb) < 2 or not np.isfinite(mse) or mse < 0:
            rows.append(
                {
                    "response": response,
                    "factor": factor,
                    "method": "tukey_hsd",
                    "group_a": a,
                    "group_b": b,
                    "n_a": len(xa),
                    "n_b": len(xb),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_difference": diff,
                    "std_error": np.nan,
                    "df": df_error,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "confidence_level": confidence_level,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "correction": "tukey_hsd",
                    "family_size": family_size,
                    "status": ResultStatus.SKIPPED.value,
                    "reason": "group_too_small",
                }
            )
            continue
        se_q = np.sqrt(0.5 * mse * (1.0 / len(xa) + 1.0 / len(xb)))
        if se_q == 0.0:
            q = np.inf if diff != 0.0 else 0.0
        else:
            q = abs(diff) / se_q
        p_value = float(studentized_range.sf(q, k, df_error))
        half = qcrit * se_q
        rows.append(
            {
                "response": response,
                "factor": factor,
                "method": "tukey_hsd",
                "group_a": a,
                "group_b": b,
                "n_a": len(xa),
                "n_b": len(xb),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_difference": diff,
                "std_error": float(se_q * np.sqrt(2.0)),
                "df": float(df_error),
                "statistic": float(q),
                "p_value": p_value,
                "confidence_level": confidence_level,
                "ci_lower": float(diff - half),
                "ci_upper": float(diff + half),
                "correction": "tukey_hsd",
                "family_size": family_size,
                "status": ResultStatus.OK.value,
                "reason": None,
            }
        )
    return rows


def _games_howell_rows(
    groups: dict[object, np.ndarray], *, response: str, factor: str, confidence_level: float
) -> list[dict]:
    levels = list(groups)
    k = len(levels)
    alpha = 1.0 - confidence_level
    family_size = k * (k - 1) // 2
    rows: list[dict] = []
    for a, b in combinations(levels, 2):
        xa, xb = groups[a], groups[b]
        mean_a, mean_b = float(xa.mean()), float(xb.mean())
        diff = mean_b - mean_a
        if len(xa) < 2 or len(xb) < 2:
            rows.append(
                {
                    "response": response,
                    "factor": factor,
                    "method": "games_howell",
                    "group_a": a,
                    "group_b": b,
                    "n_a": len(xa),
                    "n_b": len(xb),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_difference": diff,
                    "std_error": np.nan,
                    "df": np.nan,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "confidence_level": confidence_level,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "correction": "games_howell",
                    "family_size": family_size,
                    "status": ResultStatus.SKIPPED.value,
                    "reason": "group_too_small",
                }
            )
            continue
        va, vb = float(np.var(xa, ddof=1)), float(np.var(xb, ddof=1))
        aa, bb = va / len(xa), vb / len(xb)
        sum_ab = aa + bb
        if sum_ab <= 0.0:
            df = np.nan
            q = np.inf if diff != 0.0 else 0.0
            p_value = 0.0 if diff != 0.0 else 1.0
            half = 0.0
            se = 0.0
        else:
            denom = (aa**2) / (len(xa) - 1) + (bb**2) / (len(xb) - 1)
            df = (sum_ab**2) / denom if denom > 0.0 else np.inf
            se_q = np.sqrt(0.5 * sum_ab)
            q = abs(diff) / se_q
            p_value = float(studentized_range.sf(q, k, df))
            qcrit = float(studentized_range.ppf(1.0 - alpha, k, df))
            half = qcrit * se_q
            se = np.sqrt(sum_ab)
        rows.append(
            {
                "response": response,
                "factor": factor,
                "method": "games_howell",
                "group_a": a,
                "group_b": b,
                "n_a": len(xa),
                "n_b": len(xb),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_difference": diff,
                "std_error": float(se),
                "df": float(df),
                "statistic": float(q),
                "p_value": float(p_value),
                "confidence_level": confidence_level,
                "ci_lower": float(diff - half),
                "ci_upper": float(diff + half),
                "correction": "games_howell",
                "family_size": family_size,
                "status": ResultStatus.OK.value,
                "reason": None,
            }
        )
    return rows


def analyze_posthoc(
    dataset: TabularDataset,
    *,
    response: str,
    factor: str,
    methods: tuple[str, ...] = ("tukey_hsd", "games_howell"),
    confidence_level: float = 0.95,
    min_group_n: int = 2,
    max_group_levels: int = 20,
) -> PosthocResult:
    """Run explicitly requested post-hoc families without automatic method selection."""
    _validate_response_factor(dataset, response, factor)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    if min_group_n < 2:
        raise ValueError("min_group_n must be at least 2.")
    normalized = tuple(str(method).strip().lower() for method in methods)
    allowed = {"tukey_hsd", "games_howell"}
    if not normalized or any(method not in allowed for method in normalized):
        raise ValueError("methods must contain tukey_hsd and/or games_howell.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("methods cannot contain duplicates.")

    groups = _group_values(dataset, response, factor)
    if len(groups) > max_group_levels:
        raise ValueError(f"Factor {factor!r} exceeds max_group_levels={max_group_levels}.")
    rows: list[dict] = []
    if len(groups) >= 2:
        if "tukey_hsd" in normalized:
            rows.extend(
                _tukey_rows(groups, response=response, factor=factor, confidence_level=confidence_level)
            )
        if "games_howell" in normalized:
            rows.extend(
                _games_howell_rows(
                    groups, response=response, factor=factor, confidence_level=confidence_level
                )
            )
    table = pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)
    if not table.empty:
        for level, values in groups.items():
            if len(values) < min_group_n:
                mask = (table["group_a"] == level) | (table["group_b"] == level)
                table.loc[mask, "status"] = ResultStatus.SKIPPED.value
                table.loc[mask, "reason"] = "group_too_small"
                for column in ("statistic", "p_value", "ci_lower", "ci_upper"):
                    table.loc[mask, column] = np.nan
    provenance = AnalysisProvenance(
        analysis="posthoc",
        parameters={
            "response": response,
            "factor": factor,
            "methods": normalized,
            "confidence_level": confidence_level,
            "min_group_n": min_group_n,
            "max_group_levels": max_group_levels,
        },
        input_summary={"n_observations": dataset.n_observations, "n_levels": len(groups)},
    )
    return PosthocResult(comparisons=table, provenance=provenance)


__all__ = ["PosthocResult", "analyze_posthoc"]
