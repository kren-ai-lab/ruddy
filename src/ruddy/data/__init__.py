"""Domain-agnostic input data contracts."""

from ruddy.data.dataset import TabularDataset
from ruddy.data.feature_matrix import FeatureMatrix
from ruddy.data.roles import ColumnSpec, infer_column_kind
from ruddy.data.validation import AlignmentReport, align_annotations

__all__ = [
    "AlignmentReport",
    "ColumnSpec",
    "FeatureMatrix",
    "TabularDataset",
    "align_annotations",
    "infer_column_kind",
]
