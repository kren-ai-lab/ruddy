# Run multiple analyses

Use `analyze()` with `AnalysisConfig` to run several blocks on one dataset.
Each block has the same scientific behavior as its standalone function.
The default configuration runs only `profiling` and `univariate`; all other
blocks must be enabled explicitly.

## Configure a tabular run

```python
import polars as pl
from ruddy import AnalysisConfig, TabularDataset, analyze

frame = pl.DataFrame({
    "id": ["s1", "s2", "s3", "s4", "s5", "s6"],
    "activity": [1.2, 2.4, 1.8, 3.1, 2.9, 4.2],
    "family": ["A", "A", "A", "B", "B", "B"],
})
dataset = TabularDataset(
    frame,
    id_column="id",
    role_overrides={"activity": "response", "family": "factor"},
)
config = AnalysisConfig(
    enabled_blocks=("profiling", "univariate", "groups", "outliers"),
    responses=("activity",),
    groups=("family",),
    include_outlier_flags=True,
    random_state=42,
)
result = analyze(dataset, config=config)
print(result.groups.numeric_summaries)
print(result.executed_blocks)
print(result.summary())
```

Set identity and column roles when constructing the dataset. `analyze()` accepts
an existing `TabularDataset`; it does not reconstruct it from the identity/role
fields in `AnalysisConfig`.

## Available blocks and inputs

A dataset is required for every unified run. Additional inputs depend on the
selected blocks:

| Blocks | Inputs and selections |
| --- | --- |
| `profiling`, `univariate`, `bivariate`, `outliers`, `diagnostics`, `dependence`, `contingency`, `intervals`, `bayesian` | Dataset; select methods, columns and groups as applicable |
| `groups` | Dataset, response/group selections; optional aligned `annotations` |
| `pca`, `tsne`, `umap`, `multivariate`, `compositional`, `anomaly` | `features=FeatureMatrix(...)` |
| `representation` | `features` and `comparison_features` |
| `permanova` | `features` and a grouping factor in the dataset |
| `manova` | At least two selected responses and at least one factor |
| `factorial` | Response and factors/covariates, or a supported formula |
| `posthoc` | One response and one factor |
| `marginal_means` | Response and factors, or a supported formula |
| `mixed_effects` | Fixed-effects specification and `mixed_group` |

Missing required inputs raise an error. Supplying a feature matrix does not
automatically enable feature-space blocks.

## Add a feature space

```python
from ruddy import FeatureMatrix

features = FeatureMatrix(
    dataset.frame.select("activity"),
    observation_ids=dataset.observation_ids,
)
result = analyze(
    dataset,
    features=features,
    config=AnalysisConfig(
        enabled_blocks=("profiling", "pca"),
        projection_n_components=1,
        projection_scaling="standard",
        feature_alignment="strict",
        random_state=42,
    ),
)
print(result.pca.scores)
print(result.feature_alignment)
```

`feature_alignment="strict"` requires matching observation ID coverage.
`"partial"` permits incomplete coverage and reports it. This coverage check
does not subset every block to a common sample: standalone feature blocks use
the supplied feature matrix, and each analysis records its own exclusions.

Blocks use the supplied inputs independently. Enabling `pca` and `multivariate`
in one call does not feed PCA scores into multivariate analysis. For that
workflow, run PCA first, call `pca.to_feature_matrix()`, then explicitly pass
the derived feature matrix to the next analysis.

Representation comparison takes two feature spaces:

```python
# Given two FeatureMatrix objects named space_a and space_b:
result = analyze(
    dataset,
    features=space_a,
    comparison_features=space_b,
    config=AnalysisConfig(
        enabled_blocks=("representation",),
        representation_alignment="strict",
        representation_cca_components=2,
        random_state=42,
    ),
)
```

`representation_alignment` controls alignment between those two spaces,
separately from the tabular/feature coverage check.

## Find configuration options

Use `help(AnalysisConfig)` for all fields and defaults in the installed version.
The table below locates the controls most often used together.

| Analysis | Configuration fields |
| --- | --- |
| Shared | `enabled_blocks`, `responses`, `groups`, `random_state` |
| Descriptive | `quantiles`, `min_numeric_n`, `max_category_levels`, `max_missingness_patterns` |
| Bivariate/groups | `correlations`, `comparison_tests`, `p_adjust`, `min_group_n`, `pairwise` |
| Outliers | `outlier_methods`, `outlier_iqr_multiplier`, `outlier_robust_z_threshold`, `include_outlier_flags` |
| Projections | `projection_scaling`, `projection_n_components`, `projection_metric`, `tsne_*`, `umap_*` |
| Multivariate | `multivariate_*`, `mahalanobis_*` |
| MANOVA | `manova_responses`, `manova_factors`, `manova_covariates`, `manova_*` limits |
| Factorial | `factorial_response`, `factorial_factors`, `factorial_covariates`, `factorial_interactions`, `factorial_formula`, `factorial_ss_type`, `factorial_robust_covariance` |
| Dependence | `dependence_partial_covariates`, `dependence_n_permutations`, `dependence_mi_neighbors` |
| Intervals | `confidence_level`, `interval_correlations`, `bootstrap_resamples`, `bootstrap_method` |
| PERMANOVA/PERMDISP | `permanova_factor`, `permanova_metric`, `permanova_permutations` |
| Post-hoc | `posthoc_response`, `posthoc_factor`, `posthoc_methods` |
| Marginal means | `marginal_response`, `marginal_factors`, `marginal_covariates`, `marginal_interactions`, `marginal_formula`, `marginal_terms`, `marginal_p_adjust` |
| Mixed effects | `mixed_group`, `mixed_response`, `mixed_factors`, `mixed_covariates`, `mixed_formula`, `mixed_random_slopes`, `mixed_reml`, `mixed_optimizer` |
| Representation comparison | `representation_alignment`, `representation_cca_*`, `representation_distance_*`, `representation_mantel_permutations` |
| Composition | `compositional_transform`, `compositional_replace_zeros`, `compositional_zero_replacement_fraction`, `compositional_alr_denominator` |
| Bayesian EDA | `bayesian_variables`, `bayesian_groups`, `bayesian_credible_level`, `bayesian_rope`, `bayesian_draws` |
| Anomalies | `anomaly_methods`, `anomaly_scaling`, `anomaly_contamination`, `anomaly_isolation_estimators`, `anomaly_lof_neighbors` |

Names ending in `*` denote groups of fields, not literal parameter names.
To inspect defaults for a group:

```python
from dataclasses import fields

defaults = AnalysisConfig()
for field in fields(defaults):
    if field.name.startswith("factorial_"):
        print(field.name, getattr(defaults, field.name))
```

Configuration construction validates controls such as enum values, unique
blocks, quantiles, sample limits and confidence levels. Additional data-dependent
requirements are checked when the selected block runs.

Use the [method reference](METHODS.md) for assumptions and parameter meaning,
and the [results guide](RESULTS_AND_PROVENANCE.md) to inspect components,
exclusions and provenance.
