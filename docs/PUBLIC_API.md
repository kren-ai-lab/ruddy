# Public Python API inventory

This file is generated from the current installed Ruddy public namespace. Signatures reflect the code at documentation time. Detailed scientific behavior is described in `METHODS.md`.

## `Advisory`

```python
Advisory(code: 'str', message: 'str', level: 'AdvisoryLevel' = <AdvisoryLevel.WARNING: 'warning'>, context: 'Mapping[str, Any]' = <factory>) -> None
```

Non-fatal scientific diagnostic attached to a result.

## `AlignedAnnotations`

```python
AlignedAnnotations(*, source_name: 'str', data: 'pd.DataFrame', coverage: 'AnnotationCoverage', report: 'AlignmentReport | None', roles: 'Mapping[str, ColumnRole]', kinds: 'Mapping[str, ColumnKind]') -> 'None'
```

One external annotation source aligned to a base observation index.

## `AlignedRepresentationPair`

```python
AlignedRepresentationPair(x: 'np.ndarray', y: 'np.ndarray', observation_ids: 'pd.Index', source_rows_x: 'np.ndarray', source_rows_y: 'np.ndarray', exclusions: 'pd.DataFrame', alignment: 'AlignmentReport') -> None
```

AlignedRepresentationPair(x: 'np.ndarray', y: 'np.ndarray', observation_ids: 'pd.Index', source_rows_x: 'np.ndarray', source_rows_y: 'np.ndarray', exclusions: 'pd.DataFrame', alignment: 'AlignmentReport')

## `AnomalyResult`

```python
AnomalyResult(scores: 'pd.DataFrame', methods: 'pd.DataFrame', exclusions: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

AnomalyResult(scores: 'pd.DataFrame', methods: 'pd.DataFrame', exclusions: 'pd.DataFrame', provenance: 'AnalysisProvenance')

## `BayesianEDAResult`

```python
BayesianEDAResult(means: 'pd.DataFrame', mean_differences: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

BayesianEDAResult(means: 'pd.DataFrame', mean_differences: 'pd.DataFrame', provenance: 'AnalysisProvenance')

## `CCAResult`

```python
CCAResult(status: 'ResultStatus', reason: 'str | None', correlations: 'pd.DataFrame', x_weights: 'pd.DataFrame', y_weights: 'pd.DataFrame', x_loadings: 'pd.DataFrame', y_loadings: 'pd.DataFrame', x_scores: 'pd.DataFrame', y_scores: 'pd.DataFrame', advisories: 'tuple[Advisory, ...]', provenance: 'AnalysisProvenance') -> None
```

CCAResult(status: 'ResultStatus', reason: 'str | None', correlations: 'pd.DataFrame', x_weights: 'pd.DataFrame', y_weights: 'pd.DataFrame', x_loadings: 'pd.DataFrame', y_loadings: 'pd.DataFrame', x_scores: 'pd.DataFrame', y_scores: 'pd.DataFrame', advisories: 'tuple[Advisory, ...]', provenance: 'AnalysisProvenance')

## `CompositionalResult`

```python
CompositionalResult(transformed: 'FeatureMatrix', variation_matrix: 'pd.DataFrame', aitchison_distances: 'pd.DataFrame', zero_replacement: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

CompositionalResult(transformed: 'FeatureMatrix', variation_matrix: 'pd.DataFrame', aitchison_distances: 'pd.DataFrame', zero_replacement: 'pd.DataFrame', provenance: 'AnalysisProvenance')

## `RepresentationComparisonResult`

```python
RepresentationComparisonResult(cca: 'CCAResult', cka: 'pd.DataFrame', procrustes: 'pd.DataFrame', distance_similarity: 'pd.DataFrame', mantel: 'pd.DataFrame', exclusions: 'pd.DataFrame', alignment: 'AlignmentReport', provenance: 'AnalysisProvenance') -> None
```

RepresentationComparisonResult(cca: 'CCAResult', cka: 'pd.DataFrame', procrustes: 'pd.DataFrame', distance_similarity: 'pd.DataFrame', mantel: 'pd.DataFrame', exclusions: 'pd.DataFrame', alignment: 'AlignmentReport', provenance: 'AnalysisProvenance')

## `AlignmentMode`

```python
AlignmentMode(*values)
```

Policy used when aligning external annotations or metadata.

## `AnnotationCoverage`

```python
AnnotationCoverage(*values)
```

Observed coverage state for an external annotation source.

## `AnalysisBlock`

```python
AnalysisBlock(*values)
```

Top-level scientific blocks available to the unified orchestrator.

## `AnalysisConfig`

```python
AnalysisConfig(id_column: 'str | None' = None, role_overrides: 'RoleOverrides' = <factory>, kind_overrides: 'KindOverrides' = <factory>, annotation_alignment: 'AlignmentMode' = <AlignmentMode.STRICT: 'strict'>, feature_alignment: 'AlignmentMode' = <AlignmentMode.STRICT: 'strict'>, random_state: 'int' = 0, responses: 'tuple[str, ...]' = (), groups: 'tuple[str, ...]' = (), enabled_blocks: 'tuple[AnalysisBlock, ...]' = (<AnalysisBlock.PROFILING: 'profiling'>, <AnalysisBlock.UNIVARIATE: 'univariate'>), manova_responses: 'tuple[str, ...]' = (), manova_factors: 'tuple[str, ...]' = (), manova_covariates: 'tuple[str, ...]' = (), factorial_response: 'str | None' = None, factorial_factors: 'tuple[str, ...]' = (), factorial_covariates: 'tuple[str, ...]' = (), factorial_interactions: 'tuple[tuple[str, ...], ...]' = (), factorial_formula: 'str | None' = None, quantiles: 'tuple[float, ...]' = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), min_numeric_n: 'int' = 3, max_category_levels: 'int' = 50, max_missingness_patterns: 'int' = 20, max_pairwise_columns: 'int' = 200, correlations: 'tuple[CorrelationMethod, ...]' = (<CorrelationMethod.PEARSON: 'pearson'>, <CorrelationMethod.SPEARMAN: 'spearman'>, <CorrelationMethod.KENDALL: 'kendall'>), comparison_tests: 'tuple[ComparisonTest, ...]' = (<ComparisonTest.WELCH_T: 'welch_t'>, <ComparisonTest.MANN_WHITNEY: 'mann_whitney'>, <ComparisonTest.WELCH_ANOVA: 'welch_anova'>, <ComparisonTest.KRUSKAL_WALLIS: 'kruskal_wallis'>, <ComparisonTest.CHI_SQUARE: 'chi_square'>, <ComparisonTest.FISHER_EXACT: 'fisher_exact'>), p_adjust: 'PAdjustMethod' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, min_group_n: 'int' = 3, min_correlation_pairs: 'int' = 3, max_group_levels: 'int' = 20, max_correlation_columns: 'int' = 100, pairwise: 'bool' = False, outlier_methods: 'tuple[OutlierMethod, ...]' = (<OutlierMethod.IQR: 'iqr'>, <OutlierMethod.ROBUST_Z: 'robust_z'>), outlier_iqr_multiplier: 'float' = 1.5, outlier_robust_z_threshold: 'float' = 3.5, include_outlier_flags: 'bool' = False, projection_scaling: 'ScalingMethod' = <ScalingMethod.NONE: 'none'>, projection_n_components: 'int' = 2, projection_metric: 'str' = 'euclidean', tsne_perplexity: 'float' = 30.0, tsne_max_iter: 'int' = 1000, umap_n_neighbors: 'int' = 15, umap_min_dist: 'float' = 0.1, multivariate_scaling: 'ScalingMethod' = <ScalingMethod.NONE: 'none'>, multivariate_include_spearman: 'bool' = True, multivariate_max_covariance_features: 'int' = 200, multivariate_max_collinearity_features: 'int' = 100, multivariate_max_mahalanobis_features: 'int' = 100, mahalanobis_threshold_quantile: 'float' = 0.975, mahalanobis_include_robust: 'bool' = True, mahalanobis_robust_support_fraction: 'float | None' = None, manova_max_responses: 'int' = 20, manova_max_factor_levels: 'int' = 20, manova_min_level_n: 'int' = 3, factorial_ss_type: 'int' = 2, factorial_p_adjust: 'PAdjustMethod' = <PAdjustMethod.NONE: 'none'>, factorial_robust_covariance: 'str | None' = None, factorial_min_cell_n: 'int' = 2, factorial_max_factor_levels: 'int' = 20, factorial_max_design_cells: 'int' = 5000, factorial_max_design_columns: 'int' = 500, factorial_max_interaction_order: 'int' = 3, factorial_diagnostic_alpha: 'float' = 0.05, factorial_condition_number_threshold: 'float' = 30.0, distribution_max_shapiro_n: 'int' = 5000, dependence_partial_covariates: 'tuple[str, ...]' = (), dependence_min_complete_pairs: 'int' = 5, dependence_max_columns: 'int' = 50, dependence_n_permutations: 'int' = 199, dependence_mi_neighbors: 'int' = 3, confidence_level: 'float' = 0.95, interval_correlations: 'tuple[CorrelationMethod, ...]' = (<CorrelationMethod.PEARSON: 'pearson'>,), bootstrap_resamples: 'int' = 1000, bootstrap_method: 'str' = 'bca', permanova_factor: 'str | None' = None, permanova_metric: 'str' = 'euclidean', permanova_permutations: 'int' = 999, permanova_min_group_n: 'int' = 3, permanova_max_group_levels: 'int' = 20, posthoc_response: 'str | None' = None, posthoc_factor: 'str | None' = None, posthoc_methods: 'tuple[str, ...]' = ('tukey_hsd', 'games_howell'), marginal_response: 'str | None' = None, marginal_factors: 'tuple[str, ...]' = (), marginal_covariates: 'tuple[str, ...]' = (), marginal_interactions: 'tuple[tuple[str, ...], ...]' = (), marginal_formula: 'str | None' = None, marginal_terms: 'tuple[tuple[str, ...], ...]' = (), marginal_p_adjust: 'PAdjustMethod' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, mixed_group: 'str | None' = None, mixed_response: 'str | None' = None, mixed_factors: 'tuple[str, ...]' = (), mixed_covariates: 'tuple[str, ...]' = (), mixed_interactions: 'tuple[tuple[str, ...], ...]' = (), mixed_formula: 'str | None' = None, mixed_random_slopes: 'tuple[str, ...]' = (), mixed_reml: 'bool' = True, mixed_optimizer: 'str' = 'lbfgs', mixed_max_iter: 'int' = 1000, mixed_min_groups: 'int' = 3, mixed_min_group_n: 'int' = 2, representation_alignment: 'AlignmentMode' = <AlignmentMode.STRICT: 'strict'>, representation_cca_components: 'int' = 2, representation_cca_scaling: 'ScalingMethod' = <ScalingMethod.STANDARD: 'standard'>, representation_cca_max_iter: 'int' = 1000, representation_cca_tol: 'float' = 1e-06, representation_distance_metric: 'str' = 'euclidean', representation_distance_similarity_method: 'str' = 'spearman', representation_mantel_permutations: 'int' = 999, compositional_transform: 'str' = 'clr', compositional_replace_zeros: 'bool' = False, compositional_zero_replacement_fraction: 'float' = 0.65, compositional_alr_denominator: 'int' = -1, bayesian_variables: 'tuple[str, ...]' = (), bayesian_groups: 'tuple[str, ...]' = (), bayesian_credible_level: 'float' = 0.95, bayesian_rope: 'tuple[float, float]' = (-0.1, 0.1), bayesian_draws: 'int' = 5000, bayesian_min_n: 'int' = 3, anomaly_methods: 'tuple[str, ...]' = ('isolation_forest', 'lof'), anomaly_scaling: 'ScalingMethod' = <ScalingMethod.NONE: 'none'>, anomaly_contamination: 'str | float' = 'auto', anomaly_isolation_estimators: 'int' = 200, anomaly_lof_neighbors: 'int' = 20) -> None
```

Cross-cutting configuration shared by Ruddy analyses.

## `AnalysisProvenance`

```python
AnalysisProvenance(analysis: 'str', parameters: 'Mapping[str, Any]' = <factory>, input_summary: 'Mapping[str, Any]' = <factory>, random_state: 'int | None' = None, library_version: 'str' = '0.1.0.dev0', created_at: 'datetime' = <factory>) -> None
```

Minimal reproducibility record attached to scientific results.

## `AnalysisResult`

```python
AnalysisResult(status: 'ResultStatus', reason: 'str | None' = None, advisories: 'tuple[Advisory, ...]' = (), provenance: 'AnalysisProvenance | None' = None) -> None
```

Base scientific result contract used across Ruddy.

## `UnifiedAnalysisResult`

```python
UnifiedAnalysisResult(profiling: 'ProfilingResult | None' = None, univariate: 'UnivariateResult | None' = None, bivariate: 'BivariateResult | None' = None, groups: 'GroupAnalysisResult | None' = None, outliers: 'OutlierResult | None' = None, pca: 'PCAResult | None' = None, tsne: 'ProjectionResult | None' = None, umap: 'ProjectionResult | None' = None, multivariate: 'MultivariateResult | None' = None, manova: 'MANOVAResult | None' = None, factorial: 'FactorialResult | None' = None, diagnostics: 'DistributionDiagnosticsResult | None' = None, dependence: 'DependenceResult | None' = None, contingency: 'ContingencyDiagnosticsResult | None' = None, intervals: 'ConfidenceIntervalResult | None' = None, permanova: 'PermutationGroupResult | None' = None, posthoc: 'PosthocResult | None' = None, marginal_means: 'MarginalMeansResult | None' = None, mixed_effects: 'MixedEffectsResult | None' = None, representation: 'RepresentationComparisonResult | None' = None, compositional: 'CompositionalResult | None' = None, bayesian: 'BayesianEDAResult | None' = None, anomaly: 'AnomalyResult | None' = None, feature_alignment: 'AlignmentReport | None' = None, executed_blocks: 'tuple[AnalysisBlock, ...]' = (), provenance: 'AnalysisProvenance | None' = None) -> None
```

Composable graph of explicitly requested Ruddy analysis components.

## `ColumnKind`

```python
ColumnKind(*values)
```

Observed data kind used by statistical dispatch.

## `ColumnRole`

```python
ColumnRole(*values)
```

Statistical role assigned to a tabular column.

## `ComparisonTest`

```python
ComparisonTest(*values)
```

Inferential tests supported by Ruddy bivariate analysis.

## `CorrelationMethod`

```python
CorrelationMethod(*values)
```

Numeric correlation methods supported by Ruddy.

## `FeatureMatrix`

```python
FeatureMatrix(data: 'FeatureInput', *, observation_ids: 'ObservationIDs | None' = None, feature_names: 'list[str] | tuple[str, ...] | None' = None, metadata: 'pd.DataFrame | None' = None, metadata_id_column: 'str | None' = None, metadata_alignment: 'AlignmentMode | str' = <AlignmentMode.STRICT: 'strict'>, provenance: 'Mapping[str, Any] | None' = None) -> 'None'
```

Validated numeric observations × features matrix with explicit identity.

## `FactorialDesign`

```python
FactorialDesign(response: 'str', factors: 'tuple[str, ...]', covariates: 'tuple[str, ...]', interactions: 'tuple[tuple[str, ...], ...]', terms: 'tuple[FactorialTerm, ...]', requested_formula: 'str | None', resolved_formula: 'str', ss_type: 'int') -> None
```

Resolved, domain-agnostic factorial model specification.

## `FactorialResult`

```python
FactorialResult(status: 'ResultStatus', reason: 'str | None', design: 'FactorialDesign', design_terms: 'pd.DataFrame', effects: 'pd.DataFrame', coefficients: 'pd.DataFrame', diagnostics: 'pd.DataFrame', observation_diagnostics: 'pd.DataFrame', cells: 'pd.DataFrame', exclusions: 'pd.DataFrame', model_summary: 'dict[str, Any]', advisories: 'tuple[Advisory, ...]', provenance: 'AnalysisProvenance') -> None
```

Structured result for one univariate factorial ANOVA/ANCOVA model.

## `FactorialTerm`

```python
FactorialTerm(columns: 'tuple[str, ...]', kinds: 'tuple[str, ...]') -> None
```

One main or interaction term in a hierarchical factorial design.

## `MarginalMeansResult`

```python
MarginalMeansResult(means: 'pd.DataFrame', contrasts: 'pd.DataFrame', exclusions: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

Estimated marginal means plus pairwise contrasts for requested factor terms.

## `MixedEffectsResult`

```python
MixedEffectsResult(status: 'ResultStatus', reason: 'str | None', fixed_effects: 'pd.DataFrame', variance_components: 'pd.DataFrame', random_effects: 'pd.DataFrame', exclusions: 'pd.DataFrame', model_summary: 'dict[str, object]', advisories: 'tuple[Advisory, ...]', provenance: 'AnalysisProvenance') -> None
```

Structured random-intercept / numeric random-slope mixed-model result.

## `BivariateResult`

```python
BivariateResult(correlations: 'pd.DataFrame', comparisons: 'pd.DataFrame', categorical_associations: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

Complete Phase 3 mixed-type bivariate output.

## `BootstrapResult`

```python
BootstrapResult(estimate: 'float', confidence_low: 'float', confidence_high: 'float', standard_error: 'float', confidence_level: 'float', n_resamples: 'int', method: 'str', random_state: 'int', status: 'str' = 'ok', reason: 'str | None' = None) -> None
```

Compact bootstrap result without retaining the full resample distribution.

## `ConfidenceIntervalResult`

```python
ConfidenceIntervalResult(means: 'pd.DataFrame', correlations: 'pd.DataFrame', mean_differences: 'pd.DataFrame', effect_sizes: 'pd.DataFrame', odds_ratios: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

ConfidenceIntervalResult(means: 'pd.DataFrame', correlations: 'pd.DataFrame', mean_differences: 'pd.DataFrame', effect_sizes: 'pd.DataFrame', odds_ratios: 'pd.DataFrame', provenance: 'AnalysisProvenance')

## `ContingencyDiagnosticsResult`

```python
ContingencyDiagnosticsResult(summary: 'pd.DataFrame', cells: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

ContingencyDiagnosticsResult(summary: 'pd.DataFrame', cells: 'pd.DataFrame', provenance: 'AnalysisProvenance')

## `DependenceResult`

```python
DependenceResult(partial_correlations: 'pd.DataFrame', distance_correlations: 'pd.DataFrame', mutual_information: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

DependenceResult(partial_correlations: 'pd.DataFrame', distance_correlations: 'pd.DataFrame', mutual_information: 'pd.DataFrame', provenance: 'AnalysisProvenance')

## `PosthocResult`

```python
PosthocResult(comparisons: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

Structured Tukey HSD / Games-Howell comparisons.

## `DistributionDiagnosticsResult`

```python
DistributionDiagnosticsResult(normality: 'pd.DataFrame', dispersion: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

DistributionDiagnosticsResult(normality: 'pd.DataFrame', dispersion: 'pd.DataFrame', provenance: 'AnalysisProvenance')

## `GroupAnalysisResult`

```python
GroupAnalysisResult(response_catalog: 'pd.DataFrame', group_coverage: 'pd.DataFrame', numeric_summaries: 'pd.DataFrame', categorical_summaries: 'pd.DataFrame', numeric_comparisons: 'pd.DataFrame', categorical_comparisons: 'pd.DataFrame', annotation_coverage: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

Complete response-centric grouped analysis output.

## `CollinearityResult`

```python
CollinearityResult(status: 'ResultStatus', reason: 'str | None', features: 'pd.DataFrame', condition_spectrum: 'pd.DataFrame', summary: 'dict[str, Any]', exclusions: 'pd.DataFrame', preprocessing: 'dict[str, Any]', provenance: 'AnalysisProvenance') -> None
```

VIF/tolerance plus standardized design-rank diagnostics.

## `CovarianceResult`

```python
CovarianceResult(status: 'ResultStatus', reason: 'str | None', covariance: 'pd.DataFrame', pearson: 'pd.DataFrame', spearman: 'pd.DataFrame', pairwise_counts: 'pd.DataFrame', feature_diagnostics: 'pd.DataFrame', condition_spectrum: 'pd.DataFrame', summary: 'dict[str, Any]', exclusions: 'pd.DataFrame', preprocessing: 'dict[str, Any]', provenance: 'AnalysisProvenance') -> None
```

Structured covariance/correlation output for a feature matrix.

## `MANOVAResult`

```python
MANOVAResult(status: 'ResultStatus', reason: 'str | None', tests: 'pd.DataFrame', factor_levels: 'pd.DataFrame', exclusions: 'pd.DataFrame', model_summary: 'dict[str, Any]', provenance: 'AnalysisProvenance') -> None
```

Structured MANOVA statistics with complete-case accounting.

## `MahalanobisResult`

```python
MahalanobisResult(status: 'ResultStatus', reason: 'str | None', distances: 'pd.DataFrame', methods: 'pd.DataFrame', exclusions: 'pd.DataFrame', preprocessing: 'dict[str, Any]', provenance: 'AnalysisProvenance') -> None
```

Observation-level classical/robust Mahalanobis diagnostics.

## `MultivariateResult`

```python
MultivariateResult(status: 'ResultStatus', reason: 'str | None', covariance: 'CovarianceResult', collinearity: 'CollinearityResult', mahalanobis: 'MahalanobisResult', provenance: 'AnalysisProvenance') -> None
```

Combined covariance, collinearity, and multivariate-distance diagnostics.

## `PermutationGroupResult`

```python
PermutationGroupResult(status: 'ResultStatus', reason: 'str | None', summary: 'pd.DataFrame', groups: 'pd.DataFrame', distances_to_centroid: 'pd.DataFrame', exclusions: 'pd.DataFrame', alignment: 'AlignmentReport', advisories: 'tuple[Advisory, ...]', provenance: 'AnalysisProvenance') -> None
```

PERMANOVA + PERMDISP result for one grouping factor.

## `OptionalDependencyError`

```python
OptionalDependencyError
```

Raised when an optional analysis dependency is unavailable.

## `OutlierMethod`

```python
OutlierMethod(*values)
```

Univariate statistical outlier rules supported by Ruddy.

## `PAdjustMethod`

```python
PAdjustMethod(*values)
```

Multiple-testing correction policies.

## `PCAResult`

```python
PCAResult(status: 'ResultStatus', reason: 'str | None', scores: 'pd.DataFrame', loadings: 'pd.DataFrame', variance: 'pd.DataFrame', exclusions: 'pd.DataFrame', preprocessing: 'dict[str, Any]', provenance: 'AnalysisProvenance') -> None
```

Structured PCA output with scores, loadings, variance, and provenance.

## `PreparedFeatures`

```python
PreparedFeatures(matrix: 'Any', observation_ids: 'pd.Index', feature_names: 'tuple[str, ...]', source_row_indices: 'np.ndarray', exclusions: 'pd.DataFrame', metadata: 'dict[str, Any]') -> None
```

Finite-row feature matrix prepared with explicitly requested scaling.

## `ProjectionMethod`

```python
ProjectionMethod(*values)
```

Supported nonlinear exploratory projection methods.

## `ProjectionResult`

```python
ProjectionResult(status: 'ResultStatus', reason: 'str | None', method: 'ProjectionMethod', coordinates: 'pd.DataFrame', exclusions: 'pd.DataFrame', preprocessing: 'dict[str, Any]', parameters: 'dict[str, Any]', warnings: 'tuple[str, ...]', provenance: 'AnalysisProvenance', exploratory: 'bool' = True, inferential_allowed: 'bool' = False) -> None
```

Structured exploratory coordinates with explicit non-inferential semantics.

## `ProfilingResult`

```python
ProfilingResult(overview: 'dict[str, Any]', columns: 'pd.DataFrame', missingness: 'pd.DataFrame', pairwise_completeness: 'pd.DataFrame', missingness_patterns: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

Complete Phase 2 descriptive profiling output.

## `ResultStatus`

```python
ResultStatus(*values)
```

Scientific execution status for an analysis result.

## `ScalingMethod`

```python
ScalingMethod(*values)
```

Explicit feature scaling policies for projection analyses.

## `TabularDataset`

```python
TabularDataset(data: 'pd.DataFrame', *, id_column: 'str | None' = None, role_overrides: 'RoleOverrides | None' = None, kind_overrides: 'KindOverrides | None' = None) -> 'None'
```

Immutable-by-contract wrapper around a pandas DataFrame.

## `OutlierResult`

```python
OutlierResult(summaries: 'pd.DataFrame', flags: 'pd.DataFrame', quality: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

Complete Phase 5 univariate outlier and quality output.

## `UnivariateResult`

```python
UnivariateResult(profiling: 'ProfilingResult', numeric_statistics: 'pd.DataFrame', categorical_statistics: 'pd.DataFrame', categorical_frequencies: 'pd.DataFrame', datetime_statistics: 'pd.DataFrame', provenance: 'AnalysisProvenance') -> None
```

UnivariateResult(profiling: 'ProfilingResult', numeric_statistics: 'pd.DataFrame', categorical_statistics: 'pd.DataFrame', categorical_frequencies: 'pd.DataFrame', datetime_statistics: 'pd.DataFrame', provenance: 'AnalysisProvenance')

## `UnivariateTables`

```python
UnivariateTables(numeric_statistics: 'pd.DataFrame', categorical_statistics: 'pd.DataFrame', categorical_frequencies: 'pd.DataFrame', datetime_statistics: 'pd.DataFrame') -> None
```

UnivariateTables(numeric_statistics: 'pd.DataFrame', categorical_statistics: 'pd.DataFrame', categorical_frequencies: 'pd.DataFrame', datetime_statistics: 'pd.DataFrame')

## `align_annotation_source`

```python
align_annotation_source(dataset: 'TabularDataset', annotations: 'pd.DataFrame | None', *, source_name: 'str' = 'annotations', id_column: 'str | None' = None, mode: 'AlignmentMode | str' = <AlignmentMode.STRICT: 'strict'>, role_overrides: 'RoleOverrides | None' = None, kind_overrides: 'KindOverrides | None' = None) -> 'AlignedAnnotations'
```

Align one optional external annotation source to a dataset.

## `analyze`

```python
analyze(dataset: 'TabularDataset', *, config: 'AnalysisConfig | None' = None, features: 'FeatureMatrix | None' = None, comparison_features: 'FeatureMatrix | None' = None, annotations: 'Iterable[AlignedAnnotations]' = ()) -> 'UnifiedAnalysisResult'
```

Execute only the top-level analysis blocks explicitly enabled in ``config``.

## `analyze_bivariate`

```python
analyze_bivariate(dataset: 'TabularDataset', *, correlations: 'tuple[CorrelationMethod | str, ...]' = (<CorrelationMethod.PEARSON: 'pearson'>, <CorrelationMethod.SPEARMAN: 'spearman'>, <CorrelationMethod.KENDALL: 'kendall'>), comparison_tests: 'tuple[ComparisonTest | str, ...]' = (<ComparisonTest.WELCH_T: 'welch_t'>, <ComparisonTest.MANN_WHITNEY: 'mann_whitney'>, <ComparisonTest.WELCH_ANOVA: 'welch_anova'>, <ComparisonTest.KRUSKAL_WALLIS: 'kruskal_wallis'>, <ComparisonTest.CHI_SQUARE: 'chi_square'>, <ComparisonTest.FISHER_EXACT: 'fisher_exact'>), p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, min_group_n: 'int' = 3, min_correlation_pairs: 'int' = 3, max_group_levels: 'int' = 20, max_correlation_columns: 'int' = 100, max_category_levels: 'int' = 50, pairwise: 'bool' = False) -> 'BivariateResult'
```

Run the complete domain-agnostic Phase 3 bivariate layer.

## `analyze_confidence_intervals`

```python
analyze_confidence_intervals(dataset: 'TabularDataset', *, confidence_level: 'float' = 0.95, correlation_methods: 'tuple[CorrelationMethod | str, ...]' = (<CorrelationMethod.PEARSON: 'pearson'>,), bootstrap_resamples: 'int' = 1000, bootstrap_method: 'str' = 'bca', min_group_n: 'int' = 3, max_category_levels: 'int' = 20, random_state: 'int' = 0) -> 'ConfidenceIntervalResult'
```

Estimate CIs for means, correlations, binary mean differences/effects, and ORs.

## `analyze_contingency_diagnostics`

```python
analyze_contingency_diagnostics(dataset: 'TabularDataset', *, pairs: 'tuple[tuple[str, str], ...] | None' = None, max_category_levels: 'int' = 50, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>) -> 'ContingencyDiagnosticsResult'
```

Return expected counts and cell residuals for categorical associations.

## `analyze_dependence`

```python
analyze_dependence(dataset: 'TabularDataset', *, partial_covariates: 'tuple[str, ...]' = (), partial_methods: 'tuple[CorrelationMethod | str, ...]' = (<CorrelationMethod.PEARSON: 'pearson'>, <CorrelationMethod.SPEARMAN: 'spearman'>), min_complete_pairs: 'int' = 5, max_columns: 'int' = 50, n_permutations: 'int' = 199, mutual_information_neighbors: 'int' = 3, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, random_state: 'int' = 0) -> 'DependenceResult'
```

Run extended dependence analyses without interpreting dependence as causality.

## `analyze_posthoc`

```python
analyze_posthoc(dataset: 'TabularDataset', *, response: 'str', factor: 'str', methods: 'tuple[str, ...]' = ('tukey_hsd', 'games_howell'), confidence_level: 'float' = 0.95, min_group_n: 'int' = 2, max_group_levels: 'int' = 20) -> 'PosthocResult'
```

Run explicitly requested post-hoc families without automatic method selection.

## `analyze_distribution_diagnostics`

```python
analyze_distribution_diagnostics(dataset: 'TabularDataset', *, responses: 'tuple[str, ...]' = (), groups: 'tuple[str, ...]' = (), p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, min_group_n: 'int' = 2, max_group_levels: 'int' = 20, max_shapiro_n: 'int' = 5000) -> 'DistributionDiagnosticsResult'
```

Run standalone distribution diagnostics; results never select another test automatically.

## `analyze_grouped_responses`

```python
analyze_grouped_responses(dataset: 'TabularDataset', *, responses: 'Iterable[str] | None' = None, groups: 'Iterable[str] | None' = None, annotations: 'Iterable[AlignedAnnotations]' = (), comparison_tests: 'tuple[ComparisonTest | str, ...]' = (<ComparisonTest.WELCH_T: 'welch_t'>, <ComparisonTest.MANN_WHITNEY: 'mann_whitney'>, <ComparisonTest.WELCH_ANOVA: 'welch_anova'>, <ComparisonTest.KRUSKAL_WALLIS: 'kruskal_wallis'>, <ComparisonTest.CHI_SQUARE: 'chi_square'>, <ComparisonTest.FISHER_EXACT: 'fisher_exact'>), p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, min_group_n: 'int' = 3, max_group_levels: 'int' = 20, max_category_levels: 'int' = 50, pairwise: 'bool' = False) -> 'GroupAnalysisResult'
```

Analyze one or more responses independently across one or more groups.

## `analyze_groups`

```python
analyze_groups(dataset: 'TabularDataset', *, responses: 'Iterable[str] | None' = None, groups: 'Iterable[str] | None' = None, annotations: 'Iterable[AlignedAnnotations]' = (), comparison_tests: 'tuple[ComparisonTest | str, ...]' = (<ComparisonTest.WELCH_T: 'welch_t'>, <ComparisonTest.MANN_WHITNEY: 'mann_whitney'>, <ComparisonTest.WELCH_ANOVA: 'welch_anova'>, <ComparisonTest.KRUSKAL_WALLIS: 'kruskal_wallis'>, <ComparisonTest.CHI_SQUARE: 'chi_square'>, <ComparisonTest.FISHER_EXACT: 'fisher_exact'>), p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, min_group_n: 'int' = 3, max_group_levels: 'int' = 20, max_category_levels: 'int' = 50, pairwise: 'bool' = False) -> 'GroupAnalysisResult'
```

Run response-centric grouped EDA using statistical roles and data kinds.

## `analyze_factorial`

```python
analyze_factorial(dataset: 'TabularDataset', *, response: 'str | None' = None, factors: 'Iterable[str]' = (), covariates: 'Iterable[str]' = (), interactions: 'Iterable[Iterable[str]]' = (), formula: 'str | None' = None, ss_type: 'int | str' = 2, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.NONE: 'none'>, robust_covariance: 'str | None' = None, min_cell_n: 'int' = 2, max_factor_levels: 'int' = 20, max_design_cells: 'int' = 5000, max_design_columns: 'int' = 500, max_interaction_order: 'int' = 3, diagnostic_alpha: 'float' = 0.05, condition_number_threshold: 'float' = 30.0) -> 'FactorialResult'
```

Fit an explicit OLS factorial ANOVA/ANCOVA model.

## `analyze_marginal_means`

```python
analyze_marginal_means(dataset: 'TabularDataset', *, response: 'str | None' = None, factors: 'tuple[str, ...]' = (), covariates: 'tuple[str, ...]' = (), interactions: 'tuple[tuple[str, ...], ...]' = (), formula: 'str | None' = None, terms: 'tuple[str | tuple[str, ...], ...] | None' = None, confidence_level: 'float' = 0.95, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, max_interaction_order: 'int' = 3) -> 'MarginalMeansResult'
```

Estimate equal-weight marginal means and pairwise contrasts from an OLS model.

## `analyze_mixed_effects`

```python
analyze_mixed_effects(dataset: 'TabularDataset', *, group: 'str', response: 'str | None' = None, factors: 'tuple[str, ...]' = (), covariates: 'tuple[str, ...]' = (), interactions: 'tuple[tuple[str, ...], ...]' = (), formula: 'str | None' = None, random_slopes: 'tuple[str, ...]' = (), reml: 'bool' = True, optimizer: 'str' = 'lbfgs', max_iter: 'int' = 1000, confidence_level: 'float' = 0.95, min_groups: 'int' = 3, min_group_n: 'int' = 2, max_interaction_order: 'int' = 3) -> 'MixedEffectsResult'
```

Fit a random-intercept model with optional numeric random slopes.

## `build_factorial_design`

```python
build_factorial_design(dataset: 'TabularDataset', *, response: 'str | None' = None, factors: 'Iterable[str]' = (), covariates: 'Iterable[str]' = (), interactions: 'Iterable[Iterable[str]]' = (), formula: 'str | None' = None, ss_type: 'int | str' = 2, max_interaction_order: 'int' = 3) -> 'FactorialDesign'
```

Resolve a hierarchical ANOVA/ANCOVA design without evaluating arbitrary code.

## `analyze_collinearity`

```python
analyze_collinearity(features: 'FeatureMatrix', *, scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, max_features: 'int' = 100) -> 'CollinearityResult'
```

Compute VIF/tolerance with an intercept and standardized condition indices.

## `analyze_covariance_structure`

```python
analyze_covariance_structure(features: 'FeatureMatrix', *, scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, include_spearman: 'bool' = True, max_features: 'int' = 200) -> 'CovarianceResult'
```

Compute complete-case covariance/correlation structure and condition diagnostics.

## `analyze_mahalanobis`

```python
analyze_mahalanobis(features: 'FeatureMatrix', *, scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, threshold_quantile: 'float' = 0.975, include_robust: 'bool' = True, robust_support_fraction: 'float | None' = None, random_state: 'int' = 0, max_features: 'int' = 100) -> 'MahalanobisResult'
```

Compute classical and optional robust Mahalanobis distances.

## `analyze_manova`

```python
analyze_manova(dataset: 'TabularDataset', *, responses: 'Iterable[str]', factors: 'Iterable[str]', covariates: 'Iterable[str]' = (), max_responses: 'int' = 20, max_factor_levels: 'int' = 20, min_level_n: 'int' = 3) -> 'MANOVAResult'
```

Fit a main-effects MANOVA on explicitly selected responses/factors/covariates.

## `analyze_multivariate`

```python
analyze_multivariate(features: 'FeatureMatrix', *, scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, include_spearman: 'bool' = True, include_robust_mahalanobis: 'bool' = True, mahalanobis_threshold_quantile: 'float' = 0.975, robust_support_fraction: 'float | None' = None, random_state: 'int' = 0, max_covariance_features: 'int' = 200, max_collinearity_features: 'int' = 100, max_mahalanobis_features: 'int' = 100) -> 'MultivariateResult'
```

Run the core Phase-7 multivariate EDA diagnostics on one feature matrix.

## `analyze_permutation_group_structure`

```python
analyze_permutation_group_structure(features: 'FeatureMatrix', dataset: 'TabularDataset', *, factor: 'str', metric: 'str' = 'euclidean', n_permutations: 'int' = 999, random_state: 'int' = 0, min_group_n: 'int' = 3, max_group_levels: 'int' = 20, alignment: 'AlignmentMode | str' = <AlignmentMode.STRICT: 'strict'>) -> 'PermutationGroupResult'
```

Run PERMANOVA and PERMDISP together on one feature-space grouping factor.

## `attach_annotations`

```python
attach_annotations(dataset: 'TabularDataset', sources: 'Iterable[AlignedAnnotations]') -> 'TabularDataset'
```

Return a new dataset with aligned annotation columns attached.

## `bootstrap_confidence_interval`

```python
bootstrap_confidence_interval(data: 'Sequence[Sequence[float] | np.ndarray] | Sequence[float] | np.ndarray', statistic: 'Callable[..., float]', *, paired: 'bool' = False, confidence_level: 'float' = 0.95, n_resamples: 'int' = 1000, method: 'str' = 'BCa', random_state: 'int' = 0) -> 'BootstrapResult'
```

Estimate a scalar statistic and deterministic bootstrap confidence interval.

## `distance_correlation`

```python
distance_correlation(x: 'np.ndarray', y: 'np.ndarray') -> 'float'
```

Biased sample distance correlation for one-dimensional variables.

## `analyze_outliers`

```python
analyze_outliers(dataset: 'TabularDataset', *, methods: 'Iterable[OutlierMethod | str]' = (<OutlierMethod.IQR: 'iqr'>, <OutlierMethod.ROBUST_Z: 'robust_z'>), min_numeric_n: 'int' = 3, iqr_multiplier: 'float' = 1.5, robust_z_threshold: 'float' = 3.5, include_flags: 'bool' = False) -> 'OutlierResult'
```

Run non-destructive univariate outlier and numeric quality diagnostics.

## `analyze_pca`

```python
analyze_pca(features: 'FeatureMatrix', *, n_components: 'int' = 2, scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, random_state: 'int' = 0) -> 'PCAResult'
```

Run PCA with explicit scaling, row exclusion, and component provenance.

## `analyze_projection`

```python
analyze_projection(features: 'FeatureMatrix', *, method: 'ProjectionMethod | str', **kwargs: 'Any') -> 'ProjectionResult'
```

Dispatch one explicitly exploratory nonlinear projection.

## `analyze_tsne`

```python
analyze_tsne(features: 'FeatureMatrix', *, n_components: 'int' = 2, scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, metric: 'str' = 'euclidean', perplexity: 'float' = 30.0, random_state: 'int' = 0, max_iter: 'int' = 1000) -> 'ProjectionResult'
```

Run deterministic t-SNE under a fixed random seed.

## `analyze_umap`

```python
analyze_umap(features: 'FeatureMatrix', *, n_components: 'int' = 2, scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, metric: 'str' = 'euclidean', n_neighbors: 'int' = 15, min_dist: 'float' = 0.1, random_state: 'int' = 0) -> 'ProjectionResult'
```

Run UMAP when the optional ``umap-learn`` dependency is installed.

## `analyze_univariate`

```python
analyze_univariate(dataset: 'TabularDataset', *, quantiles: 'tuple[float, ...]' = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), min_numeric_n: 'int' = 3, max_category_levels: 'int' = 50, max_missingness_patterns: 'int' = 20, max_pairwise_columns: 'int' = 200) -> 'UnivariateResult'
```

Run profiling and complete univariate descriptive analysis.

## `pairwise_completeness`

```python
pairwise_completeness(dataset: 'TabularDataset', *, columns: 'tuple[str, ...] | None' = None, max_columns: 'int' = 200) -> 'pd.DataFrame'
```

Return upper-triangular pairwise non-missing completeness statistics.

## `prepare_features`

```python
prepare_features(features: 'FeatureMatrix', *, scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, minimum_observations: 'int' = 2) -> 'PreparedFeatures'
```

Exclude non-finite rows and apply only explicitly requested scaling.

## `mean_confidence_interval`

```python
mean_confidence_interval(values: 'np.ndarray', confidence_level: 'float' = 0.95) -> 'tuple[float, float, float, float]'
```

Return mean, standard error, lower CI, and upper CI using Student's t.

## `odds_ratio_confidence_interval`

```python
odds_ratio_confidence_interval(table: 'np.ndarray', confidence_level: 'float' = 0.95) -> 'tuple[float, float, float]'
```

Log-Wald interval for a 2x2 odds ratio without hidden zero-cell correction.

## `pearson_confidence_interval`

```python
pearson_confidence_interval(r: 'float', n: 'int', confidence_level: 'float' = 0.95) -> 'tuple[float, float]'
```

Fisher-z confidence interval for Pearson's correlation.

## `profile_columns`

```python
profile_columns(dataset: 'TabularDataset') -> 'pd.DataFrame'
```

Profile every dataset column without transforming source values.

## `resolve_groups`

```python
resolve_groups(dataset: 'TabularDataset', groups: 'Iterable[str] | None' = None) -> 'tuple[str, ...]'
```

Resolve grouping variables without requiring a dedicated storage type.

## `resolve_responses`

```python
resolve_responses(dataset: 'TabularDataset', responses: 'Iterable[str] | None' = None) -> 'tuple[str, ...]'
```

Resolve statistically analysable response columns.

## `profile_dataset`

```python
profile_dataset(dataset: 'TabularDataset', *, max_missingness_patterns: 'int' = 20, max_pairwise_columns: 'int' = 200) -> 'ProfilingResult'
```

Run the complete Phase 2 descriptive profiling block.

## `summarize_annotation_coverage`

```python
summarize_annotation_coverage(sources: 'Iterable[AlignedAnnotations]') -> 'pd.DataFrame'
```

Return one traceable coverage row per external annotation source.

## `summarize_categorical_associations`

```python
summarize_categorical_associations(dataset: 'TabularDataset', *, tests: 'tuple[ComparisonTest | str, ...]' = (<ComparisonTest.CHI_SQUARE: 'chi_square'>, <ComparisonTest.FISHER_EXACT: 'fisher_exact'>), pairs: 'tuple[tuple[str, str], ...] | None' = None, max_category_levels: 'int' = 50, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>) -> 'pd.DataFrame'
```

Compute selected or all unordered categorical-categorical associations.

## `summarize_categorical_statistics`

```python
summarize_categorical_statistics(dataset: 'TabularDataset', columns: 'pd.DataFrame', *, max_category_levels: 'int' = 50) -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Summarize eligible categorical, boolean, and factor variables.

## `summarize_group_coverage`

```python
summarize_group_coverage(dataset: 'TabularDataset', groups: 'Iterable[str] | None' = None, *, max_group_levels: 'int' = 20) -> 'pd.DataFrame'
```

Describe observation coverage for each grouping variable.

## `summarize_grouped_categorical_responses`

```python
summarize_grouped_categorical_responses(dataset: 'TabularDataset', responses: 'Iterable[str] | None' = None, groups: 'Iterable[str] | None' = None, *, max_group_levels: 'int' = 20) -> 'pd.DataFrame'
```

Compute within-group frequency summaries for categorical responses.

## `summarize_grouped_numeric_responses`

```python
summarize_grouped_numeric_responses(dataset: 'TabularDataset', responses: 'Iterable[str] | None' = None, groups: 'Iterable[str] | None' = None, *, max_group_levels: 'int' = 20) -> 'pd.DataFrame'
```

Compute descriptive summaries for numeric responses within groups.

## `summarize_correlations`

```python
summarize_correlations(dataset: 'TabularDataset', *, methods: 'tuple[CorrelationMethod | str, ...]' = (<CorrelationMethod.PEARSON: 'pearson'>, <CorrelationMethod.SPEARMAN: 'spearman'>, <CorrelationMethod.KENDALL: 'kendall'>), min_complete_pairs: 'int' = 3, max_columns: 'int' = 100, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>) -> 'pd.DataFrame'
```

Compute all unordered numeric-numeric correlations.

## `summarize_datetime_statistics`

```python
summarize_datetime_statistics(dataset: 'TabularDataset', columns: 'pd.DataFrame') -> 'pd.DataFrame'
```

Summarize eligible datetime variables without time-series interpretation.

## `summarize_dispersion_diagnostics`

```python
summarize_dispersion_diagnostics(dataset: 'TabularDataset', *, responses: 'tuple[str, ...]', groups: 'tuple[str, ...]', methods: 'tuple[str, ...]' = ('brown_forsythe', 'fligner_killeen'), min_group_n: 'int' = 2, max_group_levels: 'int' = 20, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>) -> 'pd.DataFrame'
```

Assess grouped dispersion using explicit robust tests.

## `summarize_general_dependence`

```python
summarize_general_dependence(dataset: 'TabularDataset', *, methods: 'tuple[str, ...]' = ('distance_correlation', 'mutual_information'), pairs: 'tuple[tuple[str, str], ...] | None' = None, min_complete_pairs: 'int' = 5, max_columns: 'int' = 50, n_permutations: 'int' = 199, mutual_information_neighbors: 'int' = 3, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>, random_state: 'int' = 0) -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Compute nonlinear/general dependence with permutation inference.

## `summarize_missingness`

```python
summarize_missingness(columns: 'pd.DataFrame') -> 'pd.DataFrame'
```

Build deterministic per-column missingness output from column profiles.

## `summarize_missingness_patterns`

```python
summarize_missingness_patterns(dataset: 'TabularDataset', *, max_patterns: 'int' = 20) -> 'pd.DataFrame'
```

Summarize common row-level missingness patterns with bounded output.

## `summarize_numeric_categorical_comparisons`

```python
summarize_numeric_categorical_comparisons(dataset: 'TabularDataset', *, tests: 'tuple[ComparisonTest | str, ...]' = (<ComparisonTest.WELCH_T: 'welch_t'>, <ComparisonTest.MANN_WHITNEY: 'mann_whitney'>, <ComparisonTest.WELCH_ANOVA: 'welch_anova'>, <ComparisonTest.KRUSKAL_WALLIS: 'kruskal_wallis'>), features: 'tuple[str, ...] | None' = None, group_columns: 'tuple[str, ...] | None' = None, min_group_n: 'int' = 3, max_group_levels: 'int' = 20, pairwise: 'bool' = False, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>) -> 'pd.DataFrame'
```

Compare selected numeric features across selected categorical group variables.

## `summarize_normality_diagnostics`

```python
summarize_normality_diagnostics(dataset: 'TabularDataset', *, methods: 'tuple[str, ...]' = ('shapiro', 'dagostino', 'anderson_darling'), columns: 'tuple[str, ...] | None' = None, max_shapiro_n: 'int' = 5000, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>) -> 'pd.DataFrame'
```

Compute explicit normality diagnostics without altering downstream test choice.

## `summarize_partial_correlations`

```python
summarize_partial_correlations(dataset: 'TabularDataset', *, covariates: 'tuple[str, ...]', methods: 'tuple[CorrelationMethod | str, ...]' = (<CorrelationMethod.PEARSON: 'pearson'>, <CorrelationMethod.SPEARMAN: 'spearman'>), pairs: 'tuple[tuple[str, str], ...] | None' = None, min_complete_pairs: 'int' = 5, max_columns: 'int' = 100, p_adjust: 'PAdjustMethod | str' = <PAdjustMethod.FDR_BH: 'fdr_bh'>) -> 'pd.DataFrame'
```

Compute Pearson/Spearman partial correlations controlling numeric covariates.

## `summarize_numeric_quality`

```python
summarize_numeric_quality(dataset: 'TabularDataset', columns: 'pd.DataFrame | None' = None, *, min_numeric_n: 'int' = 3) -> 'pd.DataFrame'
```

Summarize numeric data-quality states without changing source values.

## `summarize_numeric_statistics`

```python
summarize_numeric_statistics(dataset: 'TabularDataset', columns: 'pd.DataFrame', *, quantiles: 'tuple[float, ...]' = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), min_numeric_n: 'int' = 3) -> 'pd.DataFrame'
```

Summarize eligible numerical variables using finite observations only.

## `summarize_outliers`

```python
summarize_outliers(dataset: 'TabularDataset', columns: 'pd.DataFrame | None' = None, *, methods: 'Iterable[OutlierMethod | str]' = (<OutlierMethod.IQR: 'iqr'>, <OutlierMethod.ROBUST_Z: 'robust_z'>), min_numeric_n: 'int' = 3, iqr_multiplier: 'float' = 1.5, robust_z_threshold: 'float' = 3.5, include_flags: 'bool' = False) -> 'tuple[pd.DataFrame, pd.DataFrame]'
```

Compute deterministic univariate outlier summaries and optional row flags.

## `summarize_overview`

```python
summarize_overview(dataset: 'TabularDataset', columns: 'pd.DataFrame') -> 'dict[str, Any]'
```

Create a structured dataset overview from source data and column profiles.

## `summarize_response_catalog`

```python
summarize_response_catalog(dataset: 'TabularDataset', responses: 'Iterable[str] | None' = None) -> 'pd.DataFrame'
```

Describe selected response variables and their usable observations.

## `summarize_univariate`

```python
summarize_univariate(dataset: 'TabularDataset', columns: 'pd.DataFrame', *, quantiles: 'tuple[float, ...]' = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99), min_numeric_n: 'int' = 3, max_category_levels: 'int' = 50) -> 'UnivariateTables'
```

Compute all Phase 2 univariate descriptive tables.

## `welch_mean_difference_confidence_interval`

```python
welch_mean_difference_confidence_interval(x: 'np.ndarray', y: 'np.ndarray', confidence_level: 'float' = 0.95) -> 'tuple[float, float, float, float, float]'
```

Welch interval for mean(x) - mean(y).

## `aitchison_distance_matrix`

```python
aitchison_distance_matrix(array: 'np.ndarray', observation_ids: 'pd.Index') -> 'pd.DataFrame'
```

## `align_feature_matrices`

```python
align_feature_matrices(x: 'FeatureMatrix', y: 'FeatureMatrix', *, mode: 'AlignmentMode | str' = <AlignmentMode.STRICT: 'strict'>) -> 'AlignedRepresentationPair'
```

Align two dense feature matrices by observation identity and finite rows.

## `alr_transform`

```python
alr_transform(array: 'np.ndarray', *, denominator: 'int' = -1) -> 'np.ndarray'
```

## `analyze_anomalies`

```python
analyze_anomalies(features: 'FeatureMatrix', *, methods: 'tuple[str, ...]' = ('isolation_forest', 'lof'), scaling: 'ScalingMethod | str' = <ScalingMethod.NONE: 'none'>, contamination: 'str | float' = 'auto', isolation_estimators: 'int' = 200, lof_neighbors: 'int' = 20, random_state: 'int' = 0) -> 'AnomalyResult'
```

Score multivariate anomalies using explicitly requested ML diagnostics.

## `analyze_bayesian_eda`

```python
analyze_bayesian_eda(dataset: 'TabularDataset', *, variables: 'tuple[str, ...] | None' = None, groups: 'tuple[str, ...]' = (), credible_level: 'float' = 0.95, rope: 'tuple[float, float]' = (-0.1, 0.1), draws: 'int' = 5000, prior_mean: 'float' = 0.0, prior_kappa: 'float' = 1e-06, prior_alpha: 'float' = 1e-06, prior_beta: 'float' = 1e-06, min_n: 'int' = 3, random_state: 'int' = 0) -> 'BayesianEDAResult'
```

Estimate Bayesian means and binary-group mean differences under a weak N-IG prior.

## `analyze_composition`

```python
analyze_composition(features: 'FeatureMatrix', *, transform: 'str' = 'clr', replace_zeros: 'bool' = False, zero_replacement_fraction: 'float' = 0.65, alr_denominator: 'int' = -1) -> 'CompositionalResult'
```

Apply explicit log-ratio compositional analysis to a dense feature matrix.

## `analyze_representation_similarity`

```python
analyze_representation_similarity(x: 'FeatureMatrix', y: 'FeatureMatrix', *, alignment: 'AlignmentMode | str' = <AlignmentMode.STRICT: 'strict'>, cca_components: 'int' = 2, cca_scaling: 'ScalingMethod | str' = <ScalingMethod.STANDARD: 'standard'>, cca_max_iter: 'int' = 1000, cca_tol: 'float' = 1e-06, distance_metric: 'str' = 'euclidean', distance_similarity_method: 'str' = 'spearman', mantel_permutations: 'int' = 999, random_state: 'int' = 0) -> 'RepresentationComparisonResult'
```

Compare two numerical representation spaces for the same observations.

## `closure`

```python
closure(array: 'np.ndarray', *, total: 'float' = 1.0) -> 'np.ndarray'
```

## `clr_transform`

```python
clr_transform(array: 'np.ndarray') -> 'np.ndarray'
```

## `ilr_transform`

```python
ilr_transform(array: 'np.ndarray') -> 'np.ndarray'
```

## `linear_cka`

```python
linear_cka(x: 'np.ndarray', y: 'np.ndarray') -> 'float'
```

Linear centered-kernel alignment for two observation-aligned matrices.

## `multiplicative_zero_replacement`

```python
multiplicative_zero_replacement(array: 'np.ndarray', *, fraction: 'float' = 0.65, total: 'float' = 1.0) -> 'tuple[np.ndarray, pd.DataFrame]'
```

## `variation_matrix`

```python
variation_matrix(array: 'np.ndarray', feature_names: 'tuple[str, ...]') -> 'pd.DataFrame'
```
