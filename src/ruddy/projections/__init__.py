"""Dimensionality-reduction and projection analyses."""

from ruddy.projections.manifold import (
    ProjectionResult,
    analyze_projection,
    analyze_tsne,
    analyze_umap,
)
from ruddy.projections.pca import PCAResult, analyze_pca
from ruddy.projections.preprocessing import PreparedFeatures, prepare_features

__all__ = [
    "PCAResult",
    "PreparedFeatures",
    "ProjectionResult",
    "analyze_pca",
    "analyze_projection",
    "analyze_tsne",
    "analyze_umap",
    "prepare_features",
]
