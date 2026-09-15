"""Explicit preprocessing for high-dimensional feature projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ruddy.core.enums import ScalingMethod
from ruddy.data import FeatureMatrix


@dataclass(frozen=True, slots=True)
class PreparedFeatures:
    """Finite-row feature matrix prepared with explicitly requested scaling."""

    matrix: Any
    observation_ids: pd.Index
    feature_names: tuple[str, ...]
    source_row_indices: np.ndarray
    exclusions: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def n_observations(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def is_sparse(self) -> bool:
        return sparse.issparse(self.matrix)


def _normalize_scaling(value: ScalingMethod | str) -> ScalingMethod:
    return value if isinstance(value, ScalingMethod) else ScalingMethod(value)


def _finite_row_mask(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        csr = sparse.csr_matrix(matrix)
        mask = np.ones(csr.shape[0], dtype=bool)
        if csr.data.size:
            invalid_data = ~np.isfinite(csr.data)
            if invalid_data.any():
                coo = csr.tocoo(copy=False)
                bad_rows = np.unique(coo.row[~np.isfinite(coo.data)])
                mask[bad_rows] = False
        return mask
    array = np.asarray(matrix, dtype=np.float64)
    return np.isfinite(array).all(axis=1)


def _scale_matrix(matrix: Any, method: ScalingMethod) -> tuple[Any, dict[str, Any]]:
    is_sparse = sparse.issparse(matrix)
    if method is ScalingMethod.NONE:
        return matrix.copy(), {
            "method": method.value,
            "centering_applied": False,
            "implicit_densification": False,
        }

    if method is ScalingMethod.MINMAX and is_sparse:
        raise ValueError(
            "minmax scaling requires dense input; Ruddy will not silently densify a sparse matrix."
        )

    if method is ScalingMethod.STANDARD:
        transformer = StandardScaler(with_mean=not is_sparse)
        transformed = transformer.fit_transform(matrix)
        return transformed, {
            "method": method.value,
            "centering_applied": bool(not is_sparse),
            "implicit_densification": False,
        }

    if method is ScalingMethod.ROBUST:
        transformer = RobustScaler(with_centering=not is_sparse)
        transformed = transformer.fit_transform(matrix)
        return transformed, {
            "method": method.value,
            "centering_applied": bool(not is_sparse),
            "implicit_densification": False,
        }

    transformer = MinMaxScaler()
    transformed = transformer.fit_transform(np.asarray(matrix, dtype=np.float64))
    return np.asarray(transformed, dtype=np.float64), {
        "method": method.value,
        "centering_applied": False,
        "implicit_densification": False,
    }


def prepare_features(
    features: FeatureMatrix,
    *,
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    minimum_observations: int = 2,
) -> PreparedFeatures:
    """Exclude non-finite rows and apply only explicitly requested scaling."""
    if minimum_observations < 2:
        raise ValueError("minimum_observations must be at least 2.")
    method = _normalize_scaling(scaling)
    matrix = features.to_sparse() if features.is_sparse else features.to_array()
    valid_mask = _finite_row_mask(matrix)
    source_rows = np.flatnonzero(valid_mask).astype(np.int64)
    excluded_rows = np.flatnonzero(~valid_mask).astype(np.int64)
    ids = features.observation_ids

    exclusions = pd.DataFrame(
        {
            "source_row_index": excluded_rows,
            "observation_id": ids.take(excluded_rows).to_list(),
            "stage": "preprocessing",
            "reason": "non_finite_feature_row",
        }
    )
    if int(valid_mask.sum()) < minimum_observations:
        raise ValueError(
            "Feature projection requires at least "
            f"{minimum_observations} finite observation rows after preprocessing."
        )

    prepared = matrix[valid_mask]  # pyrefly: ignore[bad-index]
    scaled, scaling_metadata = _scale_matrix(prepared, method)
    return PreparedFeatures(
        matrix=scaled,
        observation_ids=ids.take(source_rows),
        feature_names=features.feature_names,
        source_row_indices=source_rows,
        exclusions=exclusions,
        metadata={
            "scaling": scaling_metadata,
            "n_source_observations": features.n_observations,
            "n_input_observations": int(valid_mask.sum()),
            "n_excluded_observations": int((~valid_mask).sum()),
            "n_features": features.n_features,
            "source_storage": "sparse" if features.is_sparse else "dense",
            "row_exclusion_policy": "exclude_rows_with_any_non_finite_feature",
        },
    )
