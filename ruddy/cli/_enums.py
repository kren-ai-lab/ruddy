"""CLI-layer enum choices for Typer option validation and help display."""

from __future__ import annotations

from enum import StrEnum


class CorrelationMethod(StrEnum):
    pearson = "pearson"
    spearman = "spearman"
    kendall = "kendall"


class SimilarityMethod(StrEnum):
    pearson = "pearson"
    spearman = "spearman"


class ComparisonTest(StrEnum):
    welch_t = "welch_t"
    mann_whitney = "mann_whitney"
    welch_anova = "welch_anova"
    kruskal_wallis = "kruskal_wallis"
    chi_square = "chi_square"
    fisher_exact = "fisher_exact"


class PAdjust(StrEnum):
    none = "none"
    fdr_bh = "fdr_bh"


class Scaling(StrEnum):
    none = "none"
    standard = "standard"
    robust = "robust"
    minmax = "minmax"


class Alignment(StrEnum):
    strict = "strict"
    partial = "partial"


class OutlierMethod(StrEnum):
    iqr = "iqr"
    robust_z = "robust_z"


class PosthocMethod(StrEnum):
    tukey_hsd = "tukey_hsd"
    games_howell = "games_howell"


class AnomalyMethod(StrEnum):
    isolation_forest = "isolation_forest"
    lof = "lof"


class BootstrapMethod(StrEnum):
    percentile = "percentile"
    basic = "basic"
    bca = "bca"


class RobustCovariance(StrEnum):
    none = "none"
    hc0 = "hc0"
    hc1 = "hc1"
    hc2 = "hc2"
    hc3 = "hc3"


class CompositionalTransform(StrEnum):
    clr = "clr"
    alr = "alr"
    ilr = "ilr"


class SSType(StrEnum):
    two = "2"
    three = "3"


class AnalysisBlock(StrEnum):
    profiling = "profiling"
    univariate = "univariate"
    bivariate = "bivariate"
    groups = "groups"
    outliers = "outliers"
    pca = "pca"
    tsne = "tsne"
    umap = "umap"
    multivariate = "multivariate"
    manova = "manova"
    factorial = "factorial"
    diagnostics = "diagnostics"
    dependence = "dependence"
    contingency = "contingency"
    intervals = "intervals"
    permanova = "permanova"
    posthoc = "posthoc"
    marginal_means = "marginal_means"
    mixed_effects = "mixed_effects"
    representation = "representation"
    compositional = "compositional"
    bayesian = "bayesian"
    anomaly = "anomaly"
