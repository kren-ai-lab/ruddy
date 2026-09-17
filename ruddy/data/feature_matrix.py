"""High-dimensional feature/embedding matrix contract."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scipy import sparse

from ruddy.core.enums import AlignmentMode
from ruddy.core.exceptions import FeatureMatrixValidationError
from ruddy.data.validation import (
    AlignmentReport,
    align_annotations,
    validate_observation_ids,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ruddy.core.types import FeatureInput, ObservationIDs


class FeatureMatrix:
    """Validated numeric observations x features matrix with explicit identity."""

    def __init__(
        self,
        data: FeatureInput,
        *,
        observation_ids: ObservationIDs | None = None,
        feature_names: list[str] | tuple[str, ...] | None = None,
        metadata: pd.DataFrame | None = None,
        metadata_id_column: str | None = None,
        metadata_alignment: AlignmentMode | str = AlignmentMode.STRICT,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a feature matrix from numeric input data."""
        matrix, inferred_ids, inferred_names = self._normalize_matrix(data)
        n_rows, n_features = matrix.shape

        ids = inferred_ids if observation_ids is None else observation_ids
        if ids is None:
            ids = pd.RangeIndex(n_rows)
        validated_ids = validate_observation_ids(ids)
        if len(validated_ids) != n_rows:
            msg = "observation_ids length must match the number of matrix rows."
            raise FeatureMatrixValidationError(msg)

        names = inferred_names if feature_names is None else tuple(feature_names)
        if names is None:
            names = tuple(f"feature_{index}" for index in range(n_features))
        names = tuple(str(name) for name in names)
        if len(names) != n_features:
            msg = "feature_names length must match the number of matrix columns."
            raise FeatureMatrixValidationError(msg)
        if len(set(names)) != len(names):
            msg = "feature_names must be unique."
            raise FeatureMatrixValidationError(msg)

        self._matrix = matrix
        self._observation_ids = validated_ids
        self._feature_names = names
        self._metadata: pd.DataFrame | None = None
        self._alignment_report: AlignmentReport | None = None
        self._provenance = MappingProxyType(dict(provenance or {}))

        if metadata is not None:
            aligned, report = align_annotations(
                self._observation_ids,
                metadata,
                id_column=metadata_id_column,
                mode=metadata_alignment,
            )
            self._metadata = aligned
            self._alignment_report = report

    @staticmethod
    def _normalize_matrix(
        data: FeatureInput,
    ) -> tuple[Any, pd.Index | None, tuple[str, ...] | None]:
        if isinstance(data, pd.DataFrame):
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
                msg = "FeatureMatrix DataFrames must contain only numeric columns."
                raise FeatureMatrixValidationError(msg)
            matrix = data.to_numpy(copy=True)
            return matrix, data.index.copy(), tuple(str(c) for c in data.columns)

        if sparse.issparse(data):
            matrix = cast("sparse.spmatrix", data).copy()  # pyrefly: ignore[missing-attribute]
            if matrix.ndim != 2:
                msg = "FeatureMatrix must be two-dimensional."
                raise FeatureMatrixValidationError(msg)
            if not np.issubdtype(matrix.dtype, np.number):
                msg = "FeatureMatrix must contain numeric data."
                raise FeatureMatrixValidationError(msg)
            return matrix, None, None

        matrix = np.asarray(data)
        if matrix.ndim != 2:
            msg = "FeatureMatrix must be two-dimensional."
            raise FeatureMatrixValidationError(msg)
        if not np.issubdtype(matrix.dtype, np.number):
            msg = "FeatureMatrix must contain numeric data."
            raise FeatureMatrixValidationError(msg)
        return matrix.copy(), None, None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the dimensions of the feature matrix."""
        return self._matrix.shape

    @property
    def n_observations(self) -> int:
        """Return the number of observations."""
        return self.shape[0]

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return self.shape[1]

    @property
    def observation_ids(self) -> pd.Index:
        """Return a copy of the observation identifiers."""
        return self._observation_ids.copy()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the names of the features."""
        return self._feature_names

    @property
    def is_sparse(self) -> bool:
        """Return whether the underlying matrix is sparse."""
        return sparse.issparse(self._matrix)

    @property
    def metadata(self) -> pd.DataFrame | None:
        """Return a copy of the aligned metadata, if any."""
        return None if self._metadata is None else self._metadata.copy(deep=True)

    @property
    def alignment_report(self) -> AlignmentReport | None:
        """Return the report from metadata alignment, if any."""
        return self._alignment_report

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Return immutable provenance attached to a derived feature matrix."""
        return self._provenance

    def to_array(self) -> np.ndarray:
        """Return a defensive dense copy of the matrix."""
        if sparse.issparse(self._matrix):
            return self._matrix.toarray()
        return self._matrix.copy()

    def to_sparse(self) -> sparse.spmatrix:
        """Return a defensive sparse copy of the matrix."""
        if sparse.issparse(self._matrix):
            return self._matrix.copy()
        return sparse.csr_matrix(self._matrix)

    def __len__(self) -> int:
        """Return the number of observations."""
        return self.n_observations

    def __repr__(self) -> str:
        """Return a string representation of the feature matrix."""
        return (
            f"FeatureMatrix(n_observations={self.n_observations}, "
            f"n_features={self.n_features}, sparse={self.is_sparse})"
        )
