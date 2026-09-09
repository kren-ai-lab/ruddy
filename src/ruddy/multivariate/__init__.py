"""Multivariate exploratory statistics."""

from ruddy.multivariate.analysis import MultivariateResult, analyze_multivariate
from ruddy.multivariate.collinearity import CollinearityResult, analyze_collinearity
from ruddy.multivariate.covariance import CovarianceResult, analyze_covariance_structure
from ruddy.multivariate.distances import MahalanobisResult, analyze_mahalanobis
from ruddy.multivariate.manova import MANOVAResult, analyze_manova
from ruddy.multivariate.permutation import PermutationGroupResult, analyze_permutation_group_structure

__all__ = [
    "CollinearityResult",
    "CovarianceResult",
    "MANOVAResult",
    "MahalanobisResult",
    "MultivariateResult",
    "PermutationGroupResult",
    "analyze_collinearity",
    "analyze_covariance_structure",
    "analyze_mahalanobis",
    "analyze_manova",
    "analyze_multivariate",
    "analyze_permutation_group_structure",
]
