"""Domain-agnostic input data contracts."""

from ruddy.data.annotations import AlignedAnnotations, align_annotation_source, attach_annotations
from ruddy.data.dataset import TabularDataset
from ruddy.data.feature_matrix import FeatureMatrix
from ruddy.data.roles import ColumnSpec, infer_column_kind
from ruddy.data.validation import AlignmentReport, align_annotations

__all__ = [
    "AlignedAnnotations",
    "AlignmentReport",
    "ColumnSpec",
    "FeatureMatrix",
    "TabularDataset",
    "align_annotation_source",
    "align_annotations",
    "attach_annotations",
    "infer_column_kind",
]
