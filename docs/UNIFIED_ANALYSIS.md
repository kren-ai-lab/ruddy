# Unified analysis and configuration

## `ruddy.analyze()`

The unified orchestrator is designed to compose already-implemented scientific blocks without changing their mathematics.

```python
from ruddy import AnalysisConfig, TabularDataset, FeatureMatrix, analyze

config = AnalysisConfig(
    enabled_blocks=(
        "profiling",
        "univariate",
        "bivariate",
        "groups",
        "pca",
    ),
    responses=("activity",),
    groups=("family",),
    projection_n_components=10,
    projection_scaling="standard",
    random_state=42,
)

result = analyze(
    dataset,
    config=config,
    features=features,
)
```

The function accepts:

- a required `TabularDataset`;
- an optional primary `FeatureMatrix`;
- an optional second `FeatureMatrix` used by representation comparison;
- optional aligned annotation sources;
- an optional `AnalysisConfig`.

## Default behavior

The default configuration enables only:

```text
profiling
univariate
```

This is deliberate. Running generic EDA should not silently trigger inferential tests, permutation analyses, nonlinear projections, Bayesian sampling or anomaly models.

## Available analysis blocks

```text
profiling
univariate
bivariate
groups
outliers
pca
tsne
umap
multivariate
manova
factorial
diagnostics
dependence
contingency
intervals
permanova
posthoc
marginal_means
mixed_effects
representation
compositional
bayesian
anomaly
```

Each enabled block appears in `UnifiedAnalysisResult.executed_blocks` and the corresponding result field.

## Standalone/unified parity

The unified engine delegates to the same standalone functions exposed in `ruddy.__init__`. It should therefore be treated as an execution graph, not a second implementation.

Examples:

```text
analyze_univariate(dataset)
        ↕ scientific parity
analyze(dataset, config=AnalysisConfig(enabled_blocks=("univariate",))).univariate
```

The same principle applies to bivariate, groups, PCA, multivariate, factorial, PERMANOVA, representation comparison and the other orchestrated blocks.

## Tabular + feature alignment

When feature-space and tabular analyses are combined, the configured `feature_alignment` policy determines whether the observation identities must match exactly (`strict`) or can be partially aligned (`partial`).

Alignment state is included in the unified result.

## Representation comparison

The unified orchestrator accepts a second numerical space:

```python
result = analyze(
    dataset,
    config=AnalysisConfig(enabled_blocks=("representation",)),
    features=space_a,
    comparison_features=space_b,
)
```

The representation block performs its own identity alignment according to `representation_alignment`.

## Configuration groups

`AnalysisConfig` is intentionally broad because it is the shared declarative contract for the unified engine. For standalone calls, the same parameters can be passed directly to individual functions.

### Dataset semantics

- ID column;
- role overrides;
- kind overrides;
- annotation alignment;
- feature alignment;
- global random state;
- selected responses and groups.

### Descriptive EDA

- quantiles;
- minimum numeric N;
- category-level limits;
- missingness-pattern limits;
- pairwise-completeness limits.

### Bivariate/inference

- correlation methods;
- comparison-test families;
- p-value correction;
- minimum group size;
- minimum complete pairs;
- maximum group/correlation cardinality;
- pairwise comparison toggle.

### Feature/projection

- scaling;
- component count;
- distance metric;
- t-SNE perplexity/iterations;
- UMAP neighbors/minimum distance.

### Multivariate

- covariance/VIF/Mahalanobis feature limits;
- Spearman inclusion;
- Mahalanobis threshold;
- robust covariance options;
- MANOVA limits.

### Factorial/model-based

- Type II/III SS;
- FDR policy;
- robust covariance;
- cell/level/design limits;
- maximum interaction order;
- diagnostic alpha;
- condition threshold.

### Advanced dependence/intervals

- partial-correlation covariates;
- permutation count;
- MI neighbors;
- confidence level;
- bootstrap count/method.

### Advanced group inference

- PERMANOVA factor/metric/permutations;
- post-hoc method families;
- marginal-mean model specification;
- mixed-effect model specification.

### Representation/specialized

- CCA settings;
- distance comparison settings;
- Mantel permutations;
- compositional transform/zero policy;
- Bayesian credible interval/ROPE/draws;
- anomaly methods/contamination/neighbors.

## Complete current configuration fields

The following table is generated from the current `AnalysisConfig` dataclass defaults.

| Field | Type | Default |
| --- | --- | --- |
| `id_column` | `str | None` | `None` |
| `role_overrides` | `RoleOverrides` | `mappingproxy({})` |
| `kind_overrides` | `KindOverrides` | `mappingproxy({})` |
| `annotation_alignment` | `AlignmentMode` | `'strict'` |
| `feature_alignment` | `AlignmentMode` | `'strict'` |
| `random_state` | `int` | `0` |
| `responses` | `tuple[str, ...]` | `()` |
| `groups` | `tuple[str, ...]` | `()` |
| `enabled_blocks` | `tuple[AnalysisBlock, ...]` | `('profiling', 'univariate')` |
| `manova_responses` | `tuple[str, ...]` | `()` |
| `manova_factors` | `tuple[str, ...]` | `()` |
| `manova_covariates` | `tuple[str, ...]` | `()` |
| `factorial_response` | `str | None` | `None` |
| `factorial_factors` | `tuple[str, ...]` | `()` |
| `factorial_covariates` | `tuple[str, ...]` | `()` |
| `factorial_interactions` | `tuple[tuple[str, ...], ...]` | `()` |
| `factorial_formula` | `str | None` | `None` |
| `quantiles` | `tuple[float, ...]` | `(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)` |
| `min_numeric_n` | `int` | `3` |
| `max_category_levels` | `int` | `50` |
| `max_missingness_patterns` | `int` | `20` |
| `max_pairwise_columns` | `int` | `200` |
| `correlations` | `tuple[CorrelationMethod, ...]` | `('pearson', 'spearman', 'kendall')` |
| `comparison_tests` | `tuple[ComparisonTest, ...]` | `('welch_t', 'mann_whitney', 'welch_anova', 'kruskal_wallis', 'chi_square', 'fisher_exact')` |
| `p_adjust` | `PAdjustMethod` | `'fdr_bh'` |
| `min_group_n` | `int` | `3` |
| `min_correlation_pairs` | `int` | `3` |
| `max_group_levels` | `int` | `20` |
| `max_correlation_columns` | `int` | `100` |
| `pairwise` | `bool` | `False` |
| `outlier_methods` | `tuple[OutlierMethod, ...]` | `('iqr', 'robust_z')` |
| `outlier_iqr_multiplier` | `float` | `1.5` |
| `outlier_robust_z_threshold` | `float` | `3.5` |
| `include_outlier_flags` | `bool` | `False` |
| `projection_scaling` | `ScalingMethod` | `'none'` |
| `projection_n_components` | `int` | `2` |
| `projection_metric` | `str` | `'euclidean'` |
| `tsne_perplexity` | `float` | `30.0` |
| `tsne_max_iter` | `int` | `1000` |
| `umap_n_neighbors` | `int` | `15` |
| `umap_min_dist` | `float` | `0.1` |
| `multivariate_scaling` | `ScalingMethod` | `'none'` |
| `multivariate_include_spearman` | `bool` | `True` |
| `multivariate_max_covariance_features` | `int` | `200` |
| `multivariate_max_collinearity_features` | `int` | `100` |
| `multivariate_max_mahalanobis_features` | `int` | `100` |
| `mahalanobis_threshold_quantile` | `float` | `0.975` |
| `mahalanobis_include_robust` | `bool` | `True` |
| `mahalanobis_robust_support_fraction` | `float | None` | `None` |
| `manova_max_responses` | `int` | `20` |
| `manova_max_factor_levels` | `int` | `20` |
| `manova_min_level_n` | `int` | `3` |
| `factorial_ss_type` | `int` | `2` |
| `factorial_p_adjust` | `PAdjustMethod` | `'none'` |
| `factorial_robust_covariance` | `str | None` | `None` |
| `factorial_min_cell_n` | `int` | `2` |
| `factorial_max_factor_levels` | `int` | `20` |
| `factorial_max_design_cells` | `int` | `5000` |
| `factorial_max_design_columns` | `int` | `500` |
| `factorial_max_interaction_order` | `int` | `3` |
| `factorial_diagnostic_alpha` | `float` | `0.05` |
| `factorial_condition_number_threshold` | `float` | `30.0` |
| `distribution_max_shapiro_n` | `int` | `5000` |
| `dependence_partial_covariates` | `tuple[str, ...]` | `()` |
| `dependence_min_complete_pairs` | `int` | `5` |
| `dependence_max_columns` | `int` | `50` |
| `dependence_n_permutations` | `int` | `199` |
| `dependence_mi_neighbors` | `int` | `3` |
| `confidence_level` | `float` | `0.95` |
| `interval_correlations` | `tuple[CorrelationMethod, ...]` | `('pearson')` |
| `bootstrap_resamples` | `int` | `1000` |
| `bootstrap_method` | `str` | `'bca'` |
| `permanova_factor` | `str | None` | `None` |
| `permanova_metric` | `str` | `'euclidean'` |
| `permanova_permutations` | `int` | `999` |
| `permanova_min_group_n` | `int` | `3` |
| `permanova_max_group_levels` | `int` | `20` |
| `posthoc_response` | `str | None` | `None` |
| `posthoc_factor` | `str | None` | `None` |
| `posthoc_methods` | `tuple[str, ...]` | `('tukey_hsd', 'games_howell')` |
| `marginal_response` | `str | None` | `None` |
| `marginal_factors` | `tuple[str, ...]` | `()` |
| `marginal_covariates` | `tuple[str, ...]` | `()` |
| `marginal_interactions` | `tuple[tuple[str, ...], ...]` | `()` |
| `marginal_formula` | `str | None` | `None` |
| `marginal_terms` | `tuple[tuple[str, ...], ...]` | `()` |
| `marginal_p_adjust` | `PAdjustMethod` | `'fdr_bh'` |
| `mixed_group` | `str | None` | `None` |
| `mixed_response` | `str | None` | `None` |
| `mixed_factors` | `tuple[str, ...]` | `()` |
| `mixed_covariates` | `tuple[str, ...]` | `()` |
| `mixed_interactions` | `tuple[tuple[str, ...], ...]` | `()` |
| `mixed_formula` | `str | None` | `None` |
| `mixed_random_slopes` | `tuple[str, ...]` | `()` |
| `mixed_reml` | `bool` | `True` |
| `mixed_optimizer` | `str` | `'lbfgs'` |
| `mixed_max_iter` | `int` | `1000` |
| `mixed_min_groups` | `int` | `3` |
| `mixed_min_group_n` | `int` | `2` |
| `representation_alignment` | `AlignmentMode` | `'strict'` |
| `representation_cca_components` | `int` | `2` |
| `representation_cca_scaling` | `ScalingMethod` | `'standard'` |
| `representation_cca_max_iter` | `int` | `1000` |
| `representation_cca_tol` | `float` | `1e-06` |
| `representation_distance_metric` | `str` | `'euclidean'` |
| `representation_distance_similarity_method` | `str` | `'spearman'` |
| `representation_mantel_permutations` | `int` | `999` |
| `compositional_transform` | `str` | `'clr'` |
| `compositional_replace_zeros` | `bool` | `False` |
| `compositional_zero_replacement_fraction` | `float` | `0.65` |
| `compositional_alr_denominator` | `int` | `-1` |
| `bayesian_variables` | `tuple[str, ...]` | `()` |
| `bayesian_groups` | `tuple[str, ...]` | `()` |
| `bayesian_credible_level` | `float` | `0.95` |
| `bayesian_rope` | `tuple[float, float]` | `(-0.1, 0.1)` |
| `bayesian_draws` | `int` | `5000` |
| `bayesian_min_n` | `int` | `3` |
| `anomaly_methods` | `tuple[str, ...]` | `('isolation_forest', 'lof')` |
| `anomaly_scaling` | `ScalingMethod` | `'none'` |
| `anomaly_contamination` | `str | float` | `'auto'` |
| `anomaly_isolation_estimators` | `int` | `200` |
| `anomaly_lof_neighbors` | `int` | `20` |

## Configuration validation

`AnalysisConfig` normalizes string enum values to the corresponding Ruddy enums and validates major scientific controls during construction. Examples include:

- unique enabled blocks;
- valid quantiles;
- supported scaling/correction/method enum values;
- positive sample/cardinality limits;
- valid confidence/credible probabilities;
- valid bootstrap/permutation/draw controls;
- valid interaction terms and non-duplicated model selections.

Invalid configuration should therefore fail before the orchestrator launches a partially specified analysis.

## Recommended usage pattern

For research workflows, prefer a single explicit configuration object that can be serialized/reconstructed by downstream tooling:

```python
config = AnalysisConfig(
    enabled_blocks=("profiling", "univariate", "dependence", "intervals"),
    responses=("activity",),
    groups=("family",),
    dependence_partial_covariates=("length",),
    dependence_n_permutations=999,
    bootstrap_resamples=2000,
    random_state=42,
)
```

This makes the scientific policy visible and gives the future local interface a natural configuration model to edit.
