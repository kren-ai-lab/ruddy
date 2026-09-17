"""Extended dependence analysis: partial correlation, distance correlation, and MI."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

from ruddy.core.enums import ColumnKind, ColumnRole, CorrelationMethod, PAdjustMethod
from ruddy.results import AnalysisProvenance
from ruddy.statistics import apply_multiple_testing

if TYPE_CHECKING:
    from ruddy.data import TabularDataset

PARTIAL_COLUMNS = (
    "method",
    "column_x",
    "column_y",
    "covariates",
    "n_total",
    "n_complete",
    "n_covariates",
    "coefficient",
    "statistic",
    "df",
    "p_value",
    "q_value",
    "family_id",
    "family_size",
    "correction",
    "status",
    "reason",
)
DEPENDENCE_COLUMNS = (
    "method",
    "column_x",
    "column_y",
    "column_x_kind",
    "column_y_kind",
    "n_total",
    "n_complete",
    "statistic",
    "p_value",
    "q_value",
    "family_id",
    "family_size",
    "correction",
    "n_permutations",
    "status",
    "reason",
)


@dataclass(frozen=True, slots=True)
class DependenceResult:
    partial_correlations: pd.DataFrame
    distance_correlations: pd.DataFrame
    mutual_information: pd.DataFrame
    provenance: AnalysisProvenance


def _effective_kind(dataset: TabularDataset, column: str) -> ColumnKind:
    if dataset.role_of(column) is ColumnRole.FACTOR:
        return ColumnKind.CATEGORICAL
    return dataset.kind_of(column)


def _numeric_candidates(dataset: TabularDataset) -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in dataset.schema
        if spec.kind is ColumnKind.NUMERIC
        and spec.role not in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED, ColumnRole.FACTOR}
    )


def _dependence_candidates(dataset: TabularDataset) -> tuple[str, ...]:
    allowed = {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}
    return tuple(
        spec.name
        for spec in dataset.schema
        if spec.role not in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED}
        and _effective_kind(dataset, spec.name) in allowed
    )


def _rank_columns(values: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(values, dtype=float)
    for j in range(values.shape[1]):
        ranked[:, j] = stats.rankdata(values[:, j], method="average")
    return ranked


def _residualize(y: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    design = np.column_stack([np.ones(y.shape[0]), covariates])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None, "rank_deficient_covariates"
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    if np.ptp(residuals) == 0.0:
        return None, "constant_residuals"
    return residuals, None


def summarize_partial_correlations(
    dataset: TabularDataset,
    *,
    covariates: tuple[str, ...],
    methods: tuple[CorrelationMethod | str, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
    ),
    pairs: tuple[tuple[str, str], ...] | None = None,
    min_complete_pairs: int = 5,
    max_columns: int = 100,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> pd.DataFrame:
    """Compute Pearson/Spearman partial correlations controlling numeric covariates."""
    resolved_methods = tuple(m if isinstance(m, CorrelationMethod) else CorrelationMethod(m) for m in methods)
    if any(m is CorrelationMethod.KENDALL for m in resolved_methods):
        msg = "Partial correlation currently supports Pearson and Spearman only."
        raise ValueError(msg)
    if len(set(resolved_methods)) != len(resolved_methods):
        msg = "Partial correlation methods cannot contain duplicates."
        raise ValueError(msg)
    if not covariates:
        msg = "covariates cannot be empty for partial correlation."
        raise ValueError(msg)
    numeric = _numeric_candidates(dataset)
    invalid_covariates = [c for c in covariates if c not in numeric]
    if invalid_covariates:
        msg = f"Partial-correlation covariates must be eligible numeric columns: {invalid_covariates}."
        raise ValueError(msg)
    candidates = tuple(c for c in numeric if c not in set(covariates))
    if len(candidates) > max_columns:
        msg = "Eligible partial-correlation columns exceed max_columns."
        raise ValueError(msg)
    selected_pairs = (
        tuple(combinations(candidates, 2)) if pairs is None else tuple((str(x), str(y)) for x, y in pairs)
    )
    invalid_pairs = [
        (x, y) for x, y in selected_pairs if x == y or x not in candidates or y not in candidates
    ]
    if invalid_pairs:
        msg = f"Invalid partial-correlation pairs: {invalid_pairs}."
        raise ValueError(msg)

    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []
    cov_label = ",".join(covariates)
    for method in resolved_methods:
        family_id = f"partial_correlation:{method.value}:{cov_label}"
        for x_name, y_name in selected_pairs:
            columns = (x_name, y_name, *covariates)
            matrix = frame.loc[:, columns].to_numpy(dtype=float)
            mask = np.all(np.isfinite(matrix), axis=1)
            complete = matrix[mask]
            n = int(complete.shape[0])
            row: dict[str, Any] = {
                "method": method.value,
                "column_x": x_name,
                "column_y": y_name,
                "covariates": cov_label,
                "n_total": dataset.n_observations,
                "n_complete": n,
                "n_covariates": len(covariates),
                "coefficient": np.nan,
                "statistic": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "family_id": family_id,
                "family_size": 0,
                "correction": correction.value,
                "status": "ok",
                "reason": None,
            }
            df = n - len(covariates) - 2
            if n < min_complete_pairs or df <= 0:
                row["status"] = "skipped"
                row["reason"] = "insufficient_complete_pairs"
                rows.append(row)
                continue
            if method is CorrelationMethod.SPEARMAN:
                complete = _rank_columns(complete)
            x = complete[:, 0]
            y = complete[:, 1]
            c = complete[:, 2:]
            if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
                row["status"] = "degenerate"
                row["reason"] = "constant_variable"
                rows.append(row)
                continue
            rx, reason_x = _residualize(x, c)
            ry, reason_y = _residualize(y, c)
            if rx is None or ry is None:
                row["status"] = "degenerate"
                row["reason"] = reason_x or reason_y
                rows.append(row)
                continue
            r = float(np.corrcoef(rx, ry)[0, 1])
            if not np.isfinite(r):
                row["status"] = "degenerate"
                row["reason"] = "undefined_partial_correlation"
                rows.append(row)
                continue
            if abs(r) >= 1.0:
                t_stat = float(np.sign(r) * np.inf)
                p_value = 0.0
            else:
                t_stat = float(r * np.sqrt(df / max(1e-300, 1.0 - r * r)))
                p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
            row.update(coefficient=r, statistic=t_stat, df=float(df), p_value=p_value)
            rows.append(row)
    table = pd.DataFrame(rows, columns=pd.Index(PARTIAL_COLUMNS))
    return table if table.empty else apply_multiple_testing(table, correction)


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Biased sample distance correlation for one-dimensional variables."""
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = float(np.mean(A * B))
    dvarx2 = float(np.mean(A * A))
    dvary2 = float(np.mean(B * B))
    denominator = np.sqrt(dvarx2 * dvary2)
    if denominator <= 0.0:
        return float("nan")
    return float(np.sqrt(max(0.0, dcov2) / denominator))


def _distance_permutation_pvalue(
    x: np.ndarray, y: np.ndarray, observed: float, n_permutations: int, rng: np.random.Generator
) -> float:
    extreme = 0
    for _ in range(n_permutations):
        value = distance_correlation(x, y[rng.permutation(y.size)])
        if np.isfinite(value) and value >= observed - 1e-15:
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))


def _encode_categorical(values: pd.Series) -> np.ndarray:
    return pd.Categorical(values.astype("string")).codes.astype(int)


def _mutual_information(
    x: pd.Series, y: pd.Series, x_kind: ColumnKind, y_kind: ColumnKind, *, n_neighbors: int, random_state: int
) -> float:
    if x_kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN} and y_kind in {
        ColumnKind.CATEGORICAL,
        ColumnKind.BOOLEAN,
    }:
        return float(mutual_info_score(_encode_categorical(x), _encode_categorical(y)))
    if x_kind is ColumnKind.NUMERIC and y_kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}:
        return float(
            mutual_info_classif(
                x.to_numpy(dtype=float).reshape(-1, 1),
                _encode_categorical(y),
                n_neighbors=n_neighbors,
                random_state=random_state,
            )[0]
        )
    if y_kind is ColumnKind.NUMERIC and x_kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}:
        return float(
            mutual_info_classif(
                y.to_numpy(dtype=float).reshape(-1, 1),
                _encode_categorical(x),
                n_neighbors=n_neighbors,
                random_state=random_state,
            )[0]
        )
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    xy = float(
        mutual_info_regression(
            x_arr.reshape(-1, 1), y_arr, n_neighbors=n_neighbors, random_state=random_state
        )[0]
    )
    yx = float(
        mutual_info_regression(
            y_arr.reshape(-1, 1), x_arr, n_neighbors=n_neighbors, random_state=random_state
        )[0]
    )
    return max(0.0, 0.5 * (xy + yx))


def summarize_general_dependence(
    dataset: TabularDataset,
    *,
    methods: tuple[str, ...] = ("distance_correlation", "mutual_information"),
    pairs: tuple[tuple[str, str], ...] | None = None,
    min_complete_pairs: int = 5,
    max_columns: int = 50,
    n_permutations: int = 199,
    mutual_information_neighbors: int = 3,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute nonlinear/general dependence with permutation inference."""
    methods = tuple(str(m).lower() for m in methods)
    allowed = {"distance_correlation", "mutual_information"}
    if not methods or len(set(methods)) != len(methods) or any(m not in allowed for m in methods):
        msg = "methods must be unique distance_correlation/mutual_information values."
        raise ValueError(msg)
    if n_permutations < 0:
        msg = "n_permutations cannot be negative."
        raise ValueError(msg)
    if mutual_information_neighbors < 1:
        msg = "mutual_information_neighbors must be at least 1."
        raise ValueError(msg)
    candidates = _dependence_candidates(dataset)
    if len(candidates) > max_columns:
        msg = "Eligible dependence columns exceed max_columns."
        raise ValueError(msg)
    selected_pairs = (
        tuple(combinations(candidates, 2)) if pairs is None else tuple((str(x), str(y)) for x, y in pairs)
    )
    invalid = [(x, y) for x, y in selected_pairs if x == y or x not in candidates or y not in candidates]
    if invalid:
        msg = f"Invalid dependence pairs: {invalid}."
        raise ValueError(msg)
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    frame = dataset.to_frame()
    rng = np.random.default_rng(random_state)
    outputs: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    for x_name, y_name in selected_pairs:
        x_kind = _effective_kind(dataset, x_name)
        y_kind = _effective_kind(dataset, y_name)
        pair = frame[[x_name, y_name]].dropna().copy()
        if x_kind is ColumnKind.NUMERIC:
            pair = pair[np.isfinite(pair[x_name].to_numpy(dtype=float))]
        if y_kind is ColumnKind.NUMERIC:
            pair = pair[np.isfinite(pair[y_name].to_numpy(dtype=float))]
        n = len(pair)
        for method in methods:
            row: dict[str, Any] = {
                "method": method,
                "column_x": x_name,
                "column_y": y_name,
                "column_x_kind": x_kind.value,
                "column_y_kind": y_kind.value,
                "n_total": dataset.n_observations,
                "n_complete": n,
                "statistic": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "family_id": f"dependence:{method}",
                "family_size": 0,
                "correction": correction.value,
                "n_permutations": n_permutations,
                "status": "ok",
                "reason": None,
            }
            if n < min_complete_pairs:
                row["status"] = "skipped"
                row["reason"] = "insufficient_complete_pairs"
                outputs[method].append(row)
                continue
            if method == "distance_correlation":
                if x_kind is not ColumnKind.NUMERIC or y_kind is not ColumnKind.NUMERIC:
                    row["status"] = "skipped"
                    row["reason"] = "distance_correlation_requires_numeric_pair"
                else:
                    x = pair[x_name].to_numpy(dtype=float)
                    y = pair[y_name].to_numpy(dtype=float)
                    value = distance_correlation(x, y)
                    if not np.isfinite(value):
                        row["status"] = "degenerate"
                        row["reason"] = "constant_or_undefined_distance_variance"
                    else:
                        row["statistic"] = value
                        row["p_value"] = (
                            _distance_permutation_pvalue(x, y, value, n_permutations, rng)
                            if n_permutations
                            else np.nan
                        )
            elif pair[x_name].nunique(dropna=True) < 2 or pair[y_name].nunique(dropna=True) < 2:
                row["status"] = "degenerate"
                row["reason"] = "constant_variable"
            else:
                value = _mutual_information(
                    pair[x_name],
                    pair[y_name],
                    x_kind,
                    y_kind,
                    n_neighbors=min(mutual_information_neighbors, max(1, n - 1)),
                    random_state=random_state,
                )
                row["statistic"] = value
                if n_permutations:
                    extreme = 0
                    y_values = pair[y_name].to_numpy(copy=True)
                    for permutation_index in range(n_permutations):
                        permuted = pair.copy()
                        permuted[y_name] = y_values[rng.permutation(n)]
                        permuted_value = _mutual_information(
                            permuted[x_name],
                            permuted[y_name],
                            x_kind,
                            y_kind,
                            n_neighbors=min(mutual_information_neighbors, max(1, n - 1)),
                            random_state=random_state + permutation_index + 1,
                        )
                        if permuted_value >= value - 1e-15:
                            extreme += 1
                    row["p_value"] = float((extreme + 1) / (n_permutations + 1))
            outputs[method].append(row)
    tables = []
    for method in ("distance_correlation", "mutual_information"):
        table = pd.DataFrame(outputs.get(method, []), columns=pd.Index(DEPENDENCE_COLUMNS))
        tables.append(table if table.empty else apply_multiple_testing(table, correction))
    return tables[0], tables[1]


def analyze_dependence(
    dataset: TabularDataset,
    *,
    partial_covariates: tuple[str, ...] = (),
    partial_methods: tuple[CorrelationMethod | str, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
    ),
    min_complete_pairs: int = 5,
    max_columns: int = 50,
    n_permutations: int = 199,
    mutual_information_neighbors: int = 3,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    random_state: int = 0,
) -> DependenceResult:
    """Run extended dependence analyses without interpreting dependence as causality."""
    partial = (
        summarize_partial_correlations(
            dataset,
            covariates=partial_covariates,
            methods=partial_methods,
            min_complete_pairs=min_complete_pairs,
            max_columns=max_columns,
            p_adjust=p_adjust,
        )
        if partial_covariates
        else pd.DataFrame(columns=pd.Index(PARTIAL_COLUMNS))
    )
    distance, mi = summarize_general_dependence(
        dataset,
        min_complete_pairs=min_complete_pairs,
        max_columns=max_columns,
        n_permutations=n_permutations,
        mutual_information_neighbors=mutual_information_neighbors,
        p_adjust=p_adjust,
        random_state=random_state,
    )
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    provenance = AnalysisProvenance(
        analysis="dependence",
        parameters={
            "partial_covariates": tuple(partial_covariates),
            "partial_methods": tuple(CorrelationMethod(m).value for m in partial_methods),
            "distance_correlation_estimator": "biased_double_centered",
            "mutual_information_estimator": "knn_or_empirical_by_kind",
            "n_permutations": n_permutations,
            "mutual_information_neighbors": mutual_information_neighbors,
            "p_adjust": correction.value,
            "random_state": random_state,
            "causal_interpretation": False,
        },
        input_summary={"n_observations": dataset.n_observations, "n_columns": dataset.n_columns},
        random_state=random_state,
    )
    return DependenceResult(
        partial_correlations=partial,
        distance_correlations=distance,
        mutual_information=mi,
        provenance=provenance,
    )
