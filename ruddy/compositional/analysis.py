"""Compositional-data transforms and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.spatial.distance import pdist, squareform

from ruddy.data import FeatureMatrix
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class CompositionalResult:
    transformed: FeatureMatrix
    variation_matrix: pd.DataFrame
    aitchison_distances: pd.DataFrame
    zero_replacement: pd.DataFrame
    provenance: AnalysisProvenance


def closure(array: np.ndarray, *, total: float = 1.0) -> np.ndarray:
    x = np.asarray(array, dtype=float)
    if x.ndim != 2:
        msg = "Compositional closure requires a two-dimensional matrix."
        raise ValueError(msg)
    if not np.isfinite(x).all():
        msg = "Compositions must contain only finite values."
        raise ValueError(msg)
    if (x < 0).any():
        msg = "Compositions cannot contain negative values."
        raise ValueError(msg)
    sums = x.sum(axis=1, keepdims=True)
    if (sums <= 0).any():
        msg = "Every composition must have a positive row sum."
        raise ValueError(msg)
    return x / sums * total


def multiplicative_zero_replacement(
    array: np.ndarray,
    *,
    fraction: float = 0.65,
    total: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    if not 0 < fraction < 1:
        msg = "zero_replacement_fraction must lie strictly between 0 and 1."
        raise ValueError(msg)
    x = closure(array, total=total)
    rows = []
    out = x.copy()
    for i, row in enumerate(x):
        zeros = row == 0
        m = int(zeros.sum())
        if m == 0:
            rows.append({"source_row_index": i, "zero_count": 0, "delta": 0.0, "replaced": False})
            continue
        positives = row[row > 0]
        if positives.size == 0:
            msg = "Cannot replace zeros in an all-zero composition."
            raise ValueError(msg)
        delta = float(fraction * positives.min())
        if m * delta >= total:
            msg = "Zero replacement would exhaust the composition total."
            raise ValueError(msg)
        scale = (total - m * delta) / positives.sum()
        out[i, zeros] = delta
        out[i, ~zeros] = row[~zeros] * scale
        rows.append({"source_row_index": i, "zero_count": m, "delta": delta, "replaced": True})
    return out, pd.DataFrame(rows)


def clr_transform(array: np.ndarray) -> np.ndarray:
    x = closure(array)
    if (x <= 0).any():
        msg = "CLR requires strictly positive compositions; enable explicit zero replacement first."
        raise ValueError(msg)
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True)


def alr_transform(array: np.ndarray, *, denominator: int = -1) -> np.ndarray:
    x = closure(array)
    if (x <= 0).any():
        msg = "ALR requires strictly positive compositions; enable explicit zero replacement first."
        raise ValueError(msg)
    d = x.shape[1]
    denominator = denominator % d
    keep = [i for i in range(d) if i != denominator]
    return np.log(x[:, keep] / x[:, [denominator]])


def ilr_transform(array: np.ndarray) -> np.ndarray:
    x = closure(array)
    if (x <= 0).any():
        msg = "ILR requires strictly positive compositions; enable explicit zero replacement first."
        raise ValueError(msg)
    clr = clr_transform(x)
    basis = helmert(x.shape[1], full=False).T
    return clr @ basis


def variation_matrix(array: np.ndarray, feature_names: tuple[str, ...]) -> pd.DataFrame:
    x = closure(array)
    if (x <= 0).any():
        msg = "Variation matrix requires strictly positive compositions."
        raise ValueError(msg)
    logx = np.log(x)
    d = x.shape[1]
    out = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            out[i, j] = float(np.var(logx[:, i] - logx[:, j], ddof=1)) if x.shape[0] > 1 else 0.0
    return pd.DataFrame(out, index=pd.Index(feature_names), columns=pd.Index(feature_names))


def aitchison_distance_matrix(array: np.ndarray, observation_ids: pd.Index) -> pd.DataFrame:
    clr = clr_transform(array)
    dist = squareform(pdist(clr, metric="euclidean"))
    return pd.DataFrame(dist, index=observation_ids, columns=observation_ids)


def analyze_composition(
    features: FeatureMatrix,
    *,
    transform: str = "clr",
    replace_zeros: bool = False,
    zero_replacement_fraction: float = 0.65,
    alr_denominator: int = -1,
) -> CompositionalResult:
    """Apply explicit log-ratio compositional analysis to a dense feature matrix."""
    if features.is_sparse:
        msg = "Compositional analysis currently requires dense input; Ruddy will not silently densify sparse matrices."
        raise ValueError(msg)
    raw = features.to_array().astype(float)
    if not np.isfinite(raw).all():
        msg = "Compositional analysis requires finite values in every cell."
        raise ValueError(msg)
    closed = closure(raw)
    if replace_zeros:
        prepared, replacement = multiplicative_zero_replacement(closed, fraction=zero_replacement_fraction)
    else:
        prepared = closed
        replacement = pd.DataFrame(
            {
                "source_row_index": np.arange(features.n_observations),
                "zero_count": (closed == 0).sum(axis=1),
                "delta": 0.0,
                "replaced": False,
            }
        )
        if (prepared <= 0).any():
            msg = "Log-ratio transforms require strictly positive compositions; set replace_zeros=True explicitly."
            raise ValueError(msg)
    transform = str(transform).lower()
    if transform == "clr":
        transformed_array = clr_transform(prepared)
        names = tuple(f"clr_{name}" for name in features.feature_names)
    elif transform == "alr":
        transformed_array = alr_transform(prepared, denominator=alr_denominator)
        denom = alr_denominator % features.n_features
        names = tuple(
            f"alr_{name}_over_{features.feature_names[denom]}"
            for i, name in enumerate(features.feature_names)
            if i != denom
        )
    elif transform == "ilr":
        transformed_array = ilr_transform(prepared)
        names = tuple(f"ILR{i + 1}" for i in range(transformed_array.shape[1]))
    else:
        msg = "transform must be one of: clr, alr, ilr."
        raise ValueError(msg)
    transformed = FeatureMatrix(
        transformed_array,
        observation_ids=features.observation_ids,
        feature_names=names,
        provenance={
            "derived_from": "compositional_log_ratio",
            "transform": transform,
            "closure_total": 1.0,
            "zero_replacement": replace_zeros,
            "zero_replacement_fraction": zero_replacement_fraction if replace_zeros else None,
        },
    )
    provenance = AnalysisProvenance(
        analysis="compositional",
        parameters={
            "transform": transform,
            "replace_zeros": replace_zeros,
            "zero_replacement_fraction": zero_replacement_fraction,
            "alr_denominator": alr_denominator,
        },
        input_summary={
            "n_observations": features.n_observations,
            "n_parts": features.n_features,
            "rows_with_zeros": int(((closed == 0).any(axis=1)).sum()),
        },
    )
    return CompositionalResult(
        transformed=transformed,
        variation_matrix=variation_matrix(prepared, features.feature_names),
        aitchison_distances=aitchison_distance_matrix(prepared, features.observation_ids),
        zero_replacement=replacement,
        provenance=provenance,
    )


__all__ = [
    "CompositionalResult",
    "aitchison_distance_matrix",
    "alr_transform",
    "analyze_composition",
    "closure",
    "clr_transform",
    "ilr_transform",
    "multiplicative_zero_replacement",
    "variation_matrix",
]
