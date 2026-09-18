# Results and provenance

Analysis functions return structured objects containing Polars result tables
and provenance. Some have a top-level scientific status; others report status
per table row. Inspect the status, reason and sample counts alongside estimates.

## Scientific states

| Status | Meaning | Examples |
| --- | --- | --- |
| `ok` | The requested quantity was estimable | A finite correlation on eligible pairs |
| `degenerate` | The data make the quantity non-estimable | Constant variable, zero MAD, singular covariance |
| `skipped` | A requirement was not met | Too few observations, incompatible dimensions |

A `degenerate` or `skipped` result includes a `reason`; an `ok` result has no
degeneracy reason. Missing numerical values are Polars nulls. An `ok` statistic
can still have a null p-value when inference was explicitly disabled, such as
zero-permutation execution.

Advisories describe non-fatal conditions such as unbalanced designs,
convergence warnings or negative PCoA eigenvalues. Each has a `code`, `message`,
`level` and `context`. They do not change the requested method.

## Inspect and export tables

This example is self-contained:

```python
import json
import polars as pl
from ruddy import TabularDataset, analyze_bivariate

frame = pl.DataFrame({
    "id": ["s1", "s2", "s3", "s4", "s5"],
    "x": [1.0, 2.0, 4.0, 5.0, 7.0],
    "y": [2.0, 1.0, 5.0, 4.0, 8.0],
})
result = analyze_bivariate(TabularDataset(frame, id_column="id"))
table = result.correlations
print(table.schema)  # Exact column names and dtypes for this result
print(table.filter(pl.col("status") != "ok"))
print(table.select("column_x", "column_y", "coefficient", "p_value", "q_value"))
table.write_parquet("correlations.parquet")
with open("provenance.json", "w") as output:
    json.dump(result.provenance.to_dict(), output, indent=2)
```

Tables retain declared schemas when empty or entirely null. Some columns depend
on the requested analysis, such as quantiles, feature names and component
counts. Use `.schema` and `.columns` on the returned table when consuming them.
Parquet retains column types; CSV is useful for interchange but does not encode
the full schema.

### Common inferential columns

Not every table has every column. Where present:

| Column | Interpretation |
| --- | --- |
| `status`, `reason` | Estimability and the reason for an unavailable result |
| `n_total`, `n_used`, `n_finite` | Input or eligible observation counts, depending on the method |
| `n_missing`, `n_non_finite` | Missing and non-finite counts, kept separate where reported |
| `p_value` | Unadjusted inferential p-value |
| `q_value` | Value after the configured multiple-testing correction |
| `family_id`, `family_size`, `correction` | Hypothesis family and correction policy |
| `effect_size_name`, `effect_size` | Definition and magnitude of the reported effect |

Benjamini–Hochberg correction is performed separately within each declared
family. Only `ok` rows with finite p-values belong to that inferential family.
Effect sizes and intervals can be unavailable even when other quantities in
the row are estimable; inspect the method-specific fields and diagnostics.

### Matrices and observation tables

Covariance, correlation, variation and distance matrices include row labels in
the first column: `feature` for feature matrices or `observation_id` for
observation matrices. Remaining column names are the string forms of the
corresponding labels. Coordinate/score tables similarly include observation
IDs before their component columns. Identifier columns preserve their dtype.

Exclude the label column when obtaining a numerical matrix, and align plot
metadata by ID:

```python
# Given a PCAResult named pca:
scores = pca.scores
coordinates = scores.drop("observation_id").to_numpy()
```

## Where to find each output

All major analysis results carry `provenance`. This table locates the main
outputs; method-specific interpretation is in the [method reference](METHODS.md).

| Result | Main fields |
| --- | --- |
| `ProfilingResult` | `overview`, `columns`, `missingness`, `pairwise_completeness`, `missingness_patterns` |
| `UnivariateResult` | `profiling`, `numeric_statistics`, `categorical_statistics`, `categorical_frequencies`, `datetime_statistics` |
| `DistributionDiagnosticsResult` | `normality`, `dispersion` |
| `BivariateResult` | `correlations`, `comparisons`, `categorical_associations` |
| `DependenceResult` | `partial_correlations`, `distance_correlations`, `mutual_information` |
| `ContingencyDiagnosticsResult` | `summary`, `cells` |
| `ConfidenceIntervalResult` | `means`, `correlations`, `mean_differences`, `effect_sizes`, `odds_ratios` |
| `GroupAnalysisResult` | `response_catalog`, `group_coverage`, `numeric_summaries`, `categorical_summaries`, `numeric_comparisons`, `categorical_comparisons`, `annotation_coverage` |
| `PosthocResult` | `comparisons` |
| `OutlierResult` | `summaries`, `flags`, `quality` |
| `PCAResult` | `scores`, `loadings`, `variance`, `exclusions`, `preprocessing`; `to_feature_matrix()` for derived components |
| `ProjectionResult` | `coordinates`, `exclusions`, `preprocessing`, `parameters`, `warnings`, `exploratory`, `inferential_allowed` |
| `CovarianceResult` | `covariance`, `pearson`, `spearman`, `pairwise_counts`, `feature_diagnostics`, `condition_spectrum`, `summary`, `exclusions` |
| `CollinearityResult` | `features`, `condition_spectrum`, `summary`, `exclusions` |
| `MahalanobisResult` | `distances`, `methods`, `exclusions` |
| `MultivariateResult` | `covariance`, `collinearity`, `mahalanobis` |
| `MANOVAResult` | `tests`, `factor_levels`, `exclusions`, `model_summary` |
| `PermutationGroupResult` | `summary` (PERMANOVA and PERMDISP), `groups`, `distances_to_centroid`, `exclusions`, `alignment`, `advisories` |
| `FactorialResult` | `design`, `design_terms`, `effects`, `coefficients`, `diagnostics`, `observation_diagnostics`, `cells`, `exclusions`, `model_summary`, `advisories` |
| `MarginalMeansResult` | `means`, `contrasts`, `exclusions` |
| `MixedEffectsResult` | `fixed_effects`, `variance_components`, `random_effects`, `exclusions`, `model_summary`, `advisories` |
| `RepresentationComparisonResult` | `cca`, `cka`, `procrustes`, `distance_similarity`, `mantel`, `exclusions`, `alignment` |
| `CCAResult` | `correlations`, `x_weights`, `y_weights`, `x_loadings`, `y_loadings`, `x_scores`, `y_scores`, `advisories` |
| `CompositionalResult` | `transformed` (`FeatureMatrix`), `variation_matrix`, `aitchison_distances`, `zero_replacement` |
| `BayesianEDAResult` | `means`, `mean_differences` |
| `AnomalyResult` | `scores`, `methods`, `exclusions` |

## Exclusions and alignment

Complete-case analyses return exclusions so that the analyzed sample is
traceable. Depending on the method, an exclusions table contains the observation
ID, source row position, stage and reason. Representation comparisons can report
positions from both spaces. Use observation IDs to join these records back to
source data; row positions describe their original location only.

Alignment reports record the selected mode, base/incoming counts, coverage,
missing IDs and unmatched IDs. See [data alignment](DATA_CONTRACTS.md#align-annotations-and-metadata).

## Provenance and reproducibility

`result.provenance.to_dict()` provides a serializable record:

| Field | Meaning |
| --- | --- |
| `analysis` | Analysis name |
| `parameters` | Requested parameters and policies |
| `input_summary` | Input dimensions/counts |
| `random_state` | Seed for stochastic computation |
| `library_version` | Ruddy version at execution |
| `created_at` | UTC timestamp |

Keep this record with exported results and preserve the input data, observation
IDs and dependency environment. The same data, configuration and seed reproduce
stochastic calculations under the same runtime/dependencies; timestamps will
differ between runs. Seeds apply to projections, bootstrap/permutation inference,
Bayesian draws and stochastic anomaly/robust-covariance methods.

## Unified results

`analyze()` returns `UnifiedAnalysisResult`. Enabled components are available
as attributes, through `result.component("pca")`, or in `result.components`.
`executed_blocks` identifies the components that ran; unrequested fields are
`None`. `feature_alignment` records tabular/feature alignment, and
`result.summary()` returns a serializable execution summary.

See [unified analysis](UNIFIED_ANALYSIS.md) to configure a run.
