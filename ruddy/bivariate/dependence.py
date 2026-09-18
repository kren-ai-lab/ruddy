"""Extended dependence analysis: partial correlation, distance correlation, and MI."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

from ruddy.core.enums import ColumnKind, ColumnRole, CorrelationMethod, PAdjustMethod
from ruddy.results import AnalysisProvenance
from ruddy.statistics import apply_multiple_testing
from ruddy.univariate.categorical import _category_label

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from ruddy.data import TabularDataset

PARTIAL_SCHEMA: dict[str, PolarsDataType] = {
    "method": pl.String,
    "column_x": pl.String,
    "column_y": pl.String,
    "covariates": pl.String,
    "n_total": pl.Int64,
    "n_complete": pl.Int64,
    "n_covariates": pl.Int64,
    "coefficient": pl.Float64,
    "statistic": pl.Float64,
    "df": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "correction": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
PARTIAL_COLUMNS: tuple[str, ...] = tuple(PARTIAL_SCHEMA)

DEPENDENCE_SCHEMA: dict[str, PolarsDataType] = {
    "method": pl.String,
    "column_x": pl.String,
    "column_y": pl.String,
    "column_x_kind": pl.String,
    "column_y_kind": pl.String,
    "n_total": pl.Int64,
    "n_complete": pl.Int64,
    "statistic": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "correction": pl.String,
    "n_permutations": pl.Int64,
    "status": pl.String,
    "reason": pl.String,
}
DEPENDENCE_COLUMNS: tuple[str, ...] = tuple(DEPENDENCE_SCHEMA)


@dataclass(frozen=True, slots=True)
class DependenceResult:
    """Results of bivariate dependence analyses including partial and distance correlations."""

    partial_correlations: pl.DataFrame
    distance_correlations: pl.DataFrame
    mutual_information: pl.DataFrame
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
) -> pl.DataFrame:
    """Compute Pearson/Spearman partial correlations controlling numeric covariates."""
    resolved_methods = tuple(m if isinstance(m, CorrelationMethod) else CorrelationMethod(m) for m in methods)
    if any(m is CorrelationMethod.KENDALL for m in resolved_methods):
        raise ValueError("Partial correlation currently supports Pearson and Spearman only.")
    if len(set(resolved_methods)) != len(resolved_methods):
        raise ValueError("Partial correlation methods cannot contain duplicates.")
    if not covariates:
        raise ValueError("covariates cannot be empty for partial correlation.")
    numeric = _numeric_candidates(dataset)
    invalid_covariates = [c for c in covariates if c not in numeric]
    if invalid_covariates:
        raise ValueError(
            f"Partial-correlation covariates must be eligible numeric columns: {invalid_covariates}."
        )
    candidates = tuple(c for c in numeric if c not in set(covariates))
    if len(candidates) > max_columns:
        raise ValueError("Eligible partial-correlation columns exceed max_columns.")
    selected_pairs = (
        tuple(combinations(candidates, 2)) if pairs is None else tuple((str(x), str(y)) for x, y in pairs)
    )
    invalid_pairs = [
        (x, y) for x, y in selected_pairs if x == y or x not in candidates or y not in candidates
    ]
    if invalid_pairs:
        raise ValueError(f"Invalid partial-correlation pairs: {invalid_pairs}.")

    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    frame = dataset.frame
    n_total = frame.height
    rows: list[dict[str, Any]] = []
    cov_label = ",".join(covariates)
    for method in resolved_methods:
        family_id = f"partial_correlation:{method.value}:{cov_label}"
        for x_name, y_name in selected_pairs:
            columns = (x_name, y_name, *covariates)
            matrix = frame.select(list(columns)).cast(pl.Float64).fill_null(float("nan")).to_numpy()
            mask = np.all(np.isfinite(matrix), axis=1)
            complete = matrix[mask]
            n = int(complete.shape[0])
            row: dict[str, Any] = {
                "method": method.value,
                "column_x": x_name,
                "column_y": y_name,
                "covariates": cov_label,
                "n_total": n_total,
                "n_complete": n,
                "n_covariates": len(covariates),
                "coefficient": None,
                "statistic": None,
                "df": None,
                "p_value": None,
                "q_value": None,
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
    table = pl.DataFrame(rows, schema=PARTIAL_SCHEMA)
    return table if table.height == 0 else apply_multiple_testing(table, correction)


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


def _encode_categorical(values: list[str]) -> np.ndarray:
    # Codes categorical labels by first-seen order (MI is invariant to the labeling scheme).
    mapping = {label: i for i, label in enumerate(dict.fromkeys(values))}
    return np.array([mapping[v] for v in values], dtype=int)


def _mutual_information(
    x: np.ndarray | list[str],
    y: np.ndarray | list[str],
    x_kind: ColumnKind,
    y_kind: ColumnKind,
    *,
    n_neighbors: int,
    random_state: int,
) -> float:
    x_is_cat = x_kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}
    y_is_cat = y_kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}
    if x_is_cat and y_is_cat:
        return float(mutual_info_score(_encode_categorical(list(x)), _encode_categorical(list(y))))
    if x_is_cat != y_is_cat:
        numeric = np.asarray(y if x_is_cat else x, dtype=float).reshape(-1, 1)
        codes = _encode_categorical(list(x if x_is_cat else y))
        return float(
            mutual_info_classif(numeric, codes, n_neighbors=n_neighbors, random_state=random_state)[0]
        )
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
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


def _column_valid_mask(series: pl.Series, kind: ColumnKind) -> np.ndarray:
    if kind is ColumnKind.NUMERIC:
        values = series.cast(pl.Float64).fill_null(float("nan")).to_numpy()
        return np.isfinite(values)
    raw = series.to_list()
    is_float = series.dtype.is_float()
    return np.array(
        [
            v is not None and not (is_float and isinstance(v, (float, np.floating)) and np.isnan(v))
            for v in raw
        ],
        dtype=bool,
    )


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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute nonlinear/general dependence with permutation inference."""
    methods = tuple(str(m).lower() for m in methods)
    allowed = {"distance_correlation", "mutual_information"}
    if not methods or len(set(methods)) != len(methods) or any(m not in allowed for m in methods):
        raise ValueError("methods must be unique distance_correlation/mutual_information values.")
    if n_permutations < 0:
        raise ValueError("n_permutations cannot be negative.")
    if mutual_information_neighbors < 1:
        raise ValueError("mutual_information_neighbors must be at least 1.")
    candidates = _dependence_candidates(dataset)
    if len(candidates) > max_columns:
        raise ValueError("Eligible dependence columns exceed max_columns.")
    selected_pairs = (
        tuple(combinations(candidates, 2)) if pairs is None else tuple((str(x), str(y)) for x, y in pairs)
    )
    invalid = [(x, y) for x, y in selected_pairs if x == y or x not in candidates or y not in candidates]
    if invalid:
        raise ValueError(f"Invalid dependence pairs: {invalid}.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    frame = dataset.frame
    n_total = frame.height
    rng = np.random.default_rng(random_state)
    outputs: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    for x_name, y_name in selected_pairs:
        x_kind = _effective_kind(dataset, x_name)
        y_kind = _effective_kind(dataset, y_name)
        col_x = frame.get_column(x_name)
        col_y = frame.get_column(y_name)
        mask = _column_valid_mask(col_x, x_kind) & _column_valid_mask(col_y, y_kind)
        n = int(mask.sum())

        x_data: np.ndarray | list[str]
        if x_kind is ColumnKind.NUMERIC:
            x_data = col_x.cast(pl.Float64).fill_null(float("nan")).to_numpy()[mask]
        else:
            x_raw = col_x.to_list()
            x_data = [_category_label(x_raw[i]) for i in np.flatnonzero(mask)]

        y_data: np.ndarray | list[str]
        if y_kind is ColumnKind.NUMERIC:
            y_data = col_y.cast(pl.Float64).fill_null(float("nan")).to_numpy()[mask]
        else:
            y_raw = col_y.to_list()
            y_data = [_category_label(y_raw[i]) for i in np.flatnonzero(mask)]

        for method in methods:
            row: dict[str, Any] = {
                "method": method,
                "column_x": x_name,
                "column_y": y_name,
                "column_x_kind": x_kind.value,
                "column_y_kind": y_kind.value,
                "n_total": n_total,
                "n_complete": n,
                "statistic": None,
                "p_value": None,
                "q_value": None,
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
                    x_data = np.asarray(x_data, dtype=float)
                    y_data = np.asarray(y_data, dtype=float)
                    value = distance_correlation(x_data, y_data)
                    if not np.isfinite(value):
                        row["status"] = "degenerate"
                        row["reason"] = "constant_or_undefined_distance_variance"
                    else:
                        row["statistic"] = value
                        row["p_value"] = (
                            _distance_permutation_pvalue(x_data, y_data, value, n_permutations, rng)
                            if n_permutations
                            else None
                        )
            else:
                x_unique = np.unique(x_data).size if isinstance(x_data, np.ndarray) else len(set(x_data))
                y_unique = np.unique(y_data).size if isinstance(y_data, np.ndarray) else len(set(y_data))
                if x_unique < 2 or y_unique < 2:
                    row["status"] = "degenerate"
                    row["reason"] = "constant_variable"
                else:
                    value = _mutual_information(
                        x_data,
                        y_data,
                        x_kind,
                        y_kind,
                        n_neighbors=min(mutual_information_neighbors, max(1, n - 1)),
                        random_state=random_state,
                    )
                    row["statistic"] = value
                    if n_permutations:
                        extreme = 0
                        for permutation_index in range(n_permutations):
                            perm = rng.permutation(n)
                            if isinstance(y_data, np.ndarray):
                                y_perm = y_data[perm]
                            else:
                                y_perm = [y_data[i] for i in perm]
                            permuted_value = _mutual_information(
                                x_data,
                                y_perm,
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
        table = pl.DataFrame(outputs.get(method, []), schema=DEPENDENCE_SCHEMA)
        tables.append(table if table.height == 0 else apply_multiple_testing(table, correction))
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
        else pl.DataFrame(schema=PARTIAL_SCHEMA)
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
        input_summary={"n_observations": dataset.frame.height, "n_columns": dataset.frame.width},
        random_state=random_state,
    )
    return DependenceResult(
        partial_correlations=partial,
        distance_correlations=distance,
        mutual_information=mi,
        provenance=provenance,
    )
