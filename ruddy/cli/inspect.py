"""``ruddy inspect`` — descriptive profiling and dataset diagnostics."""

from __future__ import annotations

import typer

from ruddy.cli._enums import (
    BootstrapMethod,
    CorrelationMethod,
    OutlierMethod,
    PAdjust,
    Scaling,
)
from ruddy.cli._options import (
    ANNOTATION,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_RESAMPLES,
    CATEGORICAL,
    CONFIDENCE_LEVEL,
    CONTEXT_SETTINGS,
    COVARIATE,
    DEPENDENCE_MAX_COLUMNS,
    DEPENDENCE_MIN_PAIRS,
    DEPENDENCE_PERMUTATIONS,
    EXCLUDE,
    FACTOR,
    FEATURE_COLUMN,
    FEATURES_ARGUMENT,
    GROUP,
    ID_COLUMN,
    IDS_FILE,
    INCLUDE_FLAGS,
    INTERVAL_CORRELATION,
    IQR_MULTIPLIER,
    MAHALANOBIS_QUANTILE,
    MAX_CATEGORY_LEVELS,
    MAX_COLLINEARITY_FEATURES,
    MAX_COVARIANCE_FEATURES,
    MAX_GROUP_LEVELS,
    MAX_MAHALANOBIS_FEATURES,
    MAX_MISSINGNESS_PATTERNS,
    MAX_PAIRWISE_COLUMNS,
    MAX_SHAPIRO_N,
    MI_NEIGHBORS,
    MIN_GROUP_N,
    MIN_NUMERIC_N,
    NO_ROBUST_MAHALANOBIS,
    NO_SPEARMAN,
    NUMERIC,
    OUTLIER_METHOD,
    OUTPUT_DIR,
    P_ADJUST,
    PARTIAL_COVARIATE,
    RANDOM_STATE,
    RESPONSE,
    ROBUST_SUPPORT_FRACTION,
    ROBUST_Z_THRESHOLD,
    SCALING,
    TABLE_ARGUMENT,
    bundle,
    dispatch,
)
from ruddy.cli.commands import (
    run_diagnostics,
    run_multivariate,
    run_outliers,
    run_profile,
    run_univariate,
)

app = typer.Typer(
    name="inspect",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help="Profile a dataset and inspect its distributions, outliers and structure.",
)


@app.command("profile", context_settings=CONTEXT_SETTINGS)
def profile(
    input: str = TABLE_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    response: list[str] = RESPONSE,
    factor: list[str] = FACTOR,
    covariate: list[str] = COVARIATE,
    annotation: list[str] = ANNOTATION,
    exclude: list[str] = EXCLUDE,
    numeric: list[str] = NUMERIC,
    categorical: list[str] = CATEGORICAL,
    min_numeric_n: int = MIN_NUMERIC_N,
    max_category_levels: int = MAX_CATEGORY_LEVELS,
    max_missingness_patterns: int = MAX_MISSINGNESS_PATTERNS,
    max_pairwise_columns: int = MAX_PAIRWISE_COLUMNS,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Profile a tabular dataset."""
    dispatch(run_profile, bundle(locals()), command="inspect profile")


@app.command("univariate", context_settings=CONTEXT_SETTINGS)
def univariate(
    input: str = TABLE_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    response: list[str] = RESPONSE,
    factor: list[str] = FACTOR,
    covariate: list[str] = COVARIATE,
    annotation: list[str] = ANNOTATION,
    exclude: list[str] = EXCLUDE,
    numeric: list[str] = NUMERIC,
    categorical: list[str] = CATEGORICAL,
    min_numeric_n: int = MIN_NUMERIC_N,
    max_category_levels: int = MAX_CATEGORY_LEVELS,
    max_missingness_patterns: int = MAX_MISSINGNESS_PATTERNS,
    max_pairwise_columns: int = MAX_PAIRWISE_COLUMNS,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Run profiling plus univariate descriptive statistics."""
    dispatch(run_univariate, bundle(locals()), command="inspect univariate")


@app.command("diagnostics", context_settings=CONTEXT_SETTINGS)
def diagnostics(
    input: str = TABLE_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    response: list[str] = RESPONSE,
    factor: list[str] = FACTOR,
    covariate: list[str] = COVARIATE,
    annotation: list[str] = ANNOTATION,
    exclude: list[str] = EXCLUDE,
    numeric: list[str] = NUMERIC,
    categorical: list[str] = CATEGORICAL,
    min_numeric_n: int = MIN_NUMERIC_N,
    max_category_levels: int = MAX_CATEGORY_LEVELS,
    max_missingness_patterns: int = MAX_MISSINGNESS_PATTERNS,
    max_pairwise_columns: int = MAX_PAIRWISE_COLUMNS,
    output_dir: str | None = OUTPUT_DIR,
    group: list[str] = GROUP,
    p_adjust: PAdjust = P_ADJUST,
    min_group_n: int = MIN_GROUP_N,
    max_group_levels: int = MAX_GROUP_LEVELS,
    random_state: int = RANDOM_STATE,
    max_shapiro_n: int = MAX_SHAPIRO_N,
    partial_covariate: list[str] = PARTIAL_COVARIATE,
    dependence_min_pairs: int = DEPENDENCE_MIN_PAIRS,
    dependence_max_columns: int = DEPENDENCE_MAX_COLUMNS,
    dependence_permutations: int = DEPENDENCE_PERMUTATIONS,
    mi_neighbors: int = MI_NEIGHBORS,
    confidence_level: float = CONFIDENCE_LEVEL,
    interval_correlation: list[CorrelationMethod] = INTERVAL_CORRELATION,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_method: BootstrapMethod = BOOTSTRAP_METHOD,
) -> None:
    """Run standalone normality and grouped dispersion diagnostics."""
    dispatch(run_diagnostics, bundle(locals()), command="inspect diagnostics")


@app.command("outliers", context_settings=CONTEXT_SETTINGS)
def outliers(
    input: str = TABLE_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    response: list[str] = RESPONSE,
    factor: list[str] = FACTOR,
    covariate: list[str] = COVARIATE,
    annotation: list[str] = ANNOTATION,
    exclude: list[str] = EXCLUDE,
    numeric: list[str] = NUMERIC,
    categorical: list[str] = CATEGORICAL,
    min_numeric_n: int = MIN_NUMERIC_N,
    max_category_levels: int = MAX_CATEGORY_LEVELS,
    max_missingness_patterns: int = MAX_MISSINGNESS_PATTERNS,
    max_pairwise_columns: int = MAX_PAIRWISE_COLUMNS,
    output_dir: str | None = OUTPUT_DIR,
    outlier_method: list[OutlierMethod] = OUTLIER_METHOD,
    iqr_multiplier: float = IQR_MULTIPLIER,
    robust_z_threshold: float = ROBUST_Z_THRESHOLD,
    include_flags: bool = INCLUDE_FLAGS,
) -> None:
    """Run transparent univariate outlier and numeric quality diagnostics."""
    dispatch(run_outliers, bundle(locals()), command="inspect outliers")


@app.command("multivariate", context_settings=CONTEXT_SETTINGS)
def multivariate(
    input: str = FEATURES_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    ids_file: str | None = IDS_FILE,
    feature_column: list[str] = FEATURE_COLUMN,
    scaling: Scaling = SCALING,
    no_spearman: bool = NO_SPEARMAN,
    no_robust_mahalanobis: bool = NO_ROBUST_MAHALANOBIS,
    mahalanobis_quantile: float = MAHALANOBIS_QUANTILE,
    robust_support_fraction: float | None = ROBUST_SUPPORT_FRACTION,
    random_state: int = RANDOM_STATE,
    max_covariance_features: int = MAX_COVARIANCE_FEATURES,
    max_collinearity_features: int = MAX_COLLINEARITY_FEATURES,
    max_mahalanobis_features: int = MAX_MAHALANOBIS_FEATURES,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Run covariance, collinearity, and Mahalanobis diagnostics on a feature matrix."""
    dispatch(run_multivariate, bundle(locals()), command="inspect multivariate")
