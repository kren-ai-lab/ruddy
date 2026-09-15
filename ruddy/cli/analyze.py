"""``ruddy analyze`` — the unified pipeline and the standalone analysis blocks."""

from __future__ import annotations

import typer

from ruddy.cli._enums import (
    Alignment,
    AnomalyMethod,
    BootstrapMethod,
    ComparisonTest,
    CompositionalTransform,
    CorrelationMethod,
    PAdjust,
    PosthocMethod,
    Scaling,
    SimilarityMethod,
)
from ruddy.cli._options import (
    ALR_DENOMINATOR,
    ANNOTATION,
    ANNOTATION_ALIGNMENT,
    ANNOTATION_CATEGORICAL,
    ANNOTATION_COVARIATE,
    ANNOTATION_FACTOR,
    ANNOTATION_FILE,
    ANNOTATION_ID_COLUMN,
    ANNOTATION_NAME,
    ANNOTATION_NUMERIC,
    ANNOTATION_RESPONSE,
    ANOMALY_METHOD,
    ANOMALY_SCALING,
    BAYESIAN_CREDIBLE_LEVEL,
    BAYESIAN_DRAWS,
    BAYESIAN_GROUP,
    BAYESIAN_MIN_N,
    BAYESIAN_ROPE_HIGH,
    BAYESIAN_ROPE_LOW,
    BAYESIAN_VARIABLE,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_RESAMPLES,
    CATEGORICAL,
    CCA_COMPONENTS,
    CCA_MAX_ITER,
    CCA_SCALING,
    CCA_TOL,
    COMPARISON_TEST,
    COMPOSITIONAL_TRANSFORM,
    CONFIDENCE_LEVEL,
    CONTAMINATION,
    CONTEXT_SETTINGS,
    CORRELATION,
    COVARIATE,
    DEPENDENCE_MAX_COLUMNS,
    DEPENDENCE_MIN_PAIRS,
    DEPENDENCE_PERMUTATIONS,
    DISTANCE_METRIC,
    DISTANCE_SIMILARITY_METHOD,
    EXCLUDE,
    FACTOR,
    FEATURE_COLUMN,
    FEATURES_ARGUMENT,
    GROUP,
    ID_COLUMN,
    IDS_FILE,
    INTERVAL_CORRELATION,
    ISOLATION_ESTIMATORS,
    LOF_NEIGHBORS,
    MANTEL_PERMUTATIONS,
    MAX_CATEGORY_LEVELS,
    MAX_CORRELATION_COLUMNS,
    MAX_GROUP_LEVELS,
    MAX_MISSINGNESS_PATTERNS,
    MAX_PAIRWISE_COLUMNS,
    MAX_SHAPIRO_N,
    MI_NEIGHBORS,
    MIN_CORRELATION_PAIRS,
    MIN_GROUP_N,
    MIN_GROUP_N_2,
    MIN_NUMERIC_N,
    NUMERIC,
    OUTPUT_DIR,
    P_ADJUST,
    PAIRWISE,
    PARTIAL_COVARIATE,
    POSTHOC_FACTOR_REQUIRED,
    POSTHOC_METHOD,
    POSTHOC_RESPONSE_REQUIRED,
    RANDOM_STATE,
    REPLACE_ZEROS,
    REPRESENTATION_ALIGNMENT,
    REPRESENTATION_X_ARGUMENT,
    REPRESENTATION_Y_ARGUMENT,
    RESPONSE,
    TABLE_ARGUMENT,
    X_FEATURE_COLUMN,
    X_ID_COLUMN,
    X_IDS_FILE,
    Y_FEATURE_COLUMN,
    Y_ID_COLUMN,
    Y_IDS_FILE,
    ZERO_REPLACEMENT_FRACTION,
    bundle,
    dispatch,
)
from ruddy.cli.commands import (
    run_anomaly,
    run_bayesian,
    run_bivariate,
    run_compositional,
    run_contingency,
    run_dependence,
    run_groups,
    run_intervals,
    run_posthoc,
    run_representation,
)

app = typer.Typer(
    name="analyze",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help="Run a single explicitly configured analysis block.",
)


@app.command("bivariate", context_settings=CONTEXT_SETTINGS)
def bivariate(
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
    correlation: list[CorrelationMethod] = CORRELATION,
    comparison_test: list[ComparisonTest] = COMPARISON_TEST,
    min_group_n: int = MIN_GROUP_N,
    min_correlation_pairs: int = MIN_CORRELATION_PAIRS,
    max_group_levels: int = MAX_GROUP_LEVELS,
    max_correlation_columns: int = MAX_CORRELATION_COLUMNS,
    pairwise: bool = PAIRWISE,
    p_adjust: PAdjust = P_ADJUST,
) -> None:
    """Run mixed-type bivariate associations and inference."""
    dispatch(run_bivariate, bundle(locals()), command="analyze bivariate")


@app.command("dependence", context_settings=CONTEXT_SETTINGS)
def dependence(
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
    p_adjust: PAdjust = P_ADJUST,
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
    """Run partial, distance-correlation, and mutual-information analysis."""
    dispatch(run_dependence, bundle(locals()), command="analyze dependence")


@app.command("contingency", context_settings=CONTEXT_SETTINGS)
def contingency(
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
    p_adjust: PAdjust = P_ADJUST,
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
    """Run cell-level contingency diagnostics."""
    dispatch(run_contingency, bundle(locals()), command="analyze contingency")


@app.command("intervals", context_settings=CONTEXT_SETTINGS)
def intervals(
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
    min_group_n: int = MIN_GROUP_N,
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
    """Estimate confidence intervals for common EDA estimands."""
    dispatch(run_intervals, bundle(locals()), command="analyze intervals")


@app.command("groups", context_settings=CONTEXT_SETTINGS)
def groups(
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
    min_group_n: int = MIN_GROUP_N,
    max_group_levels: int = MAX_GROUP_LEVELS,
    comparison_test: list[ComparisonTest] = COMPARISON_TEST,
    p_adjust: PAdjust = P_ADJUST,
    pairwise: bool = PAIRWISE,
    annotation_file: str | None = ANNOTATION_FILE,
    annotation_name: str = ANNOTATION_NAME,
    annotation_id_column: str | None = ANNOTATION_ID_COLUMN,
    annotation_alignment: Alignment = ANNOTATION_ALIGNMENT,
    annotation_factor: list[str] = ANNOTATION_FACTOR,
    annotation_response: list[str] = ANNOTATION_RESPONSE,
    annotation_covariate: list[str] = ANNOTATION_COVARIATE,
    annotation_numeric: list[str] = ANNOTATION_NUMERIC,
    annotation_categorical: list[str] = ANNOTATION_CATEGORICAL,
) -> None:
    """Analyze one or more statistical responses across grouping variables."""
    dispatch(run_groups, bundle(locals()), command="analyze groups")


@app.command("posthoc", context_settings=CONTEXT_SETTINGS)
def posthoc(
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
    posthoc_response: str = POSTHOC_RESPONSE_REQUIRED,
    posthoc_factor: str = POSTHOC_FACTOR_REQUIRED,
    posthoc_method: list[PosthocMethod] = POSTHOC_METHOD,
    min_group_n: int = MIN_GROUP_N_2,
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
    """Run explicitly requested Tukey HSD and/or Games-Howell comparisons."""
    dispatch(run_posthoc, bundle(locals()), command="analyze posthoc")


@app.command("representation", context_settings=CONTEXT_SETTINGS)
def representation(
    input_x: str = REPRESENTATION_X_ARGUMENT,
    input_y: str = REPRESENTATION_Y_ARGUMENT,
    x_id_column: str | None = X_ID_COLUMN,
    y_id_column: str | None = Y_ID_COLUMN,
    x_ids_file: str | None = X_IDS_FILE,
    y_ids_file: str | None = Y_IDS_FILE,
    x_feature_column: list[str] = X_FEATURE_COLUMN,
    y_feature_column: list[str] = Y_FEATURE_COLUMN,
    representation_alignment: Alignment = REPRESENTATION_ALIGNMENT,
    cca_components: int = CCA_COMPONENTS,
    cca_scaling: Scaling = CCA_SCALING,
    cca_max_iter: int = CCA_MAX_ITER,
    cca_tol: float = CCA_TOL,
    distance_metric: str = DISTANCE_METRIC,
    distance_similarity_method: SimilarityMethod = DISTANCE_SIMILARITY_METHOD,
    mantel_permutations: int = MANTEL_PERMUTATIONS,
    random_state: int = RANDOM_STATE,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Compare two aligned numerical representation spaces."""
    dispatch(run_representation, bundle(locals()), command="analyze representation")


@app.command("compositional", context_settings=CONTEXT_SETTINGS)
def compositional(
    input: str = FEATURES_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    ids_file: str | None = IDS_FILE,
    feature_column: list[str] = FEATURE_COLUMN,
    compositional_transform: CompositionalTransform = COMPOSITIONAL_TRANSFORM,
    replace_zeros: bool = REPLACE_ZEROS,
    zero_replacement_fraction: float = ZERO_REPLACEMENT_FRACTION,
    alr_denominator: int = ALR_DENOMINATOR,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Run explicit compositional log-ratio analysis."""
    dispatch(run_compositional, bundle(locals()), command="analyze compositional")


@app.command("bayesian", context_settings=CONTEXT_SETTINGS)
def bayesian(
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
    bayesian_variable: list[str] = BAYESIAN_VARIABLE,
    bayesian_group: list[str] = BAYESIAN_GROUP,
    bayesian_credible_level: float = BAYESIAN_CREDIBLE_LEVEL,
    bayesian_rope_low: float = BAYESIAN_ROPE_LOW,
    bayesian_rope_high: float = BAYESIAN_ROPE_HIGH,
    bayesian_draws: int = BAYESIAN_DRAWS,
    bayesian_min_n: int = BAYESIAN_MIN_N,
    random_state: int = RANDOM_STATE,
) -> None:
    """Run scoped Bayesian exploratory estimation."""
    dispatch(run_bayesian, bundle(locals()), command="analyze bayesian")


@app.command("anomaly", context_settings=CONTEXT_SETTINGS)
def anomaly(
    input: str = FEATURES_ARGUMENT,
    id_column: str | None = ID_COLUMN,
    ids_file: str | None = IDS_FILE,
    feature_column: list[str] = FEATURE_COLUMN,
    anomaly_method: list[AnomalyMethod] = ANOMALY_METHOD,
    anomaly_scaling: Scaling = ANOMALY_SCALING,
    contamination: str = CONTAMINATION,
    isolation_estimators: int = ISOLATION_ESTIMATORS,
    lof_neighbors: int = LOF_NEIGHBORS,
    random_state: int = RANDOM_STATE,
    output_dir: str | None = OUTPUT_DIR,
) -> None:
    """Score multivariate anomalies without modifying observations."""
    dispatch(run_anomaly, bundle(locals()), command="analyze anomaly")
