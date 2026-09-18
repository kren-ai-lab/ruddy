# Results, scientific states and provenance

Ruddy is designed around structured results that preserve not only computed numbers but also whether those numbers were scientifically estimable and how they were obtained.

## Scientific status contract

The core status enum is:

```text
ok
degenerate
skipped
```

### `ok`

The requested statistic/model was estimable under the method's current requirements.

An `ok` result cannot carry a degeneracy reason.

### `degenerate`

The method is conceptually applicable, but the observed data make the requested quantity degenerate or non-estimable. Examples include:

- constant variables;
- zero MAD for modified Z-scores;
- zero-rank representation;
- singular covariance;
- rank-deficient model design;
- constant response;
- undefined statistic/effect size.

A degenerate result must carry an explicit `reason`.

### `skipped`

The requested analysis did not satisfy an implementation/mathematical prerequisite and was intentionally not run. Examples include:

- insufficient observations;
- group too small;
- Shapiro sample above configured limit;
- requested CCA components above effective rank;
- Procrustes with unequal dimensionality;
- LOF neighbor count incompatible with sample size.

A skipped result must carry an explicit `reason`.

## `AnalysisResult`

The generic base contract contains:

```python
AnalysisResult(
    status=...,
    reason=...,
    advisories=...,
    provenance=...,
)
```

Convenience constructors enforce valid state combinations:

```python
AnalysisResult.ok()
AnalysisResult.degenerate("constant")
AnalysisResult.skipped("insufficient_observations")
```

## Advisories

An `Advisory` is a non-fatal scientific condition.

```python
Advisory(
    code="unbalanced_factorial_design",
    message="...",
    level="warning",
    context={...},
)
```

Advisories are appropriate when an analysis can still be returned but the result deserves explicit attention.

Examples currently used in the codebase include:

- unbalanced factorial design;
- empty/small factorial cells;
- Type II SS with interactions;
- model/inference warnings;
- flagged factorial diagnostics;
- influence diagnostics;
- negative PCoA eigenvalues;
- singular random-effect covariance;
- mixed-model convergence/fit warnings;
- CCA fit failure details.

The existence of an advisory does not automatically alter the requested method.

## `AnalysisProvenance`

Each major result carries a reproducibility record:

```python
AnalysisProvenance(
    analysis="pca",
    parameters={...},
    input_summary={...},
    random_state=42,
    library_version="0.1.0.dev0",
    created_at=...,
)
```

Fields:

| Field | Purpose |
| --- | --- |
| `analysis` | Stable name of the executed analysis |
| `parameters` | Parameters/policies used by that analysis |
| `input_summary` | Key dimensions/counts describing the input actually analyzed |
| `random_state` | Seed where stochastic computation is involved |
| `library_version` | Ruddy version at execution time |
| `created_at` | UTC timestamp |

`parameters` and `input_summary` are stored as immutable mappings inside the result object.

## Exclusion tables

Methods that operate on a complete set of finite rows generally return an explicit exclusions table.

A typical schema includes:

```text
source_row_index
observation_id
stage
reason
```

Representation alignment may additionally include source-row positions from both spaces.

Common exclusion reasons include:

```text
non_finite_feature_row
missing_or_non_finite_model_value
missing_group_value
```

The point is not merely debugging: a downstream interface can communicate exactly how many observations contributed to each analysis.

## Alignment reports

`AlignmentReport` stores:

- mode (`strict`/`partial`);
- base count;
- incoming/annotation count;
- covered count;
- coverage fraction;
- missing IDs;
- unmatched IDs;
- boolean complete state.

This is used for external metadata and representation comparisons.

## Major result objects

### `ProfilingResult`

```text
overview
columns
missingness
pairwise_completeness
missingness_patterns
provenance
```

### `UnivariateResult`

```text
profiling
numeric_statistics
categorical_statistics
categorical_frequencies
datetime_statistics
provenance
```

### `BivariateResult`

Contains the main correlation, numeric/categorical comparison and categorical-association tables plus provenance.

### `DependenceResult`

```text
partial_correlations
distance_correlations
mutual_information
provenance
```

### `GroupAnalysisResult`

```text
response_catalog
group_coverage
numeric_summaries
categorical_summaries
numeric_comparisons
categorical_comparisons
annotation_coverage
provenance
```

### `OutlierResult`

```text
summaries
flags
quality
provenance
```

### `PCAResult`

```text
status
reason
scores
loadings
variance
exclusions
preprocessing
provenance
```

The object can create a derived principal-component `FeatureMatrix`.

### `ProjectionResult`

```text
status
reason
method
coordinates
exclusions
preprocessing
parameters
warnings
provenance
exploratory
inferential_allowed
```

### `CovarianceResult`

```text
status
reason
covariance
pearson
spearman
pairwise_counts
feature_diagnostics
condition_spectrum
summary
exclusions
preprocessing
provenance
```

### `CollinearityResult`

```text
status
reason
features
condition_spectrum
summary
exclusions
preprocessing
provenance
```

### `MahalanobisResult`

```text
status
reason
distances
methods
exclusions
preprocessing
provenance
```

### `MultivariateResult`

Aggregates covariance, collinearity and Mahalanobis results with a top-level status/provenance.

### `MANOVAResult`

```text
status
reason
tests
factor_levels
exclusions
model_summary
provenance
```

### `PermutationGroupResult`

```text
status
reason
summary
groups
distances_to_centroid
exclusions
alignment
advisories
provenance
```

The summary contains both PERMANOVA and PERMDISP rows.

### `FactorialResult`

```text
status
reason
design
design_terms
effects
coefficients
diagnostics
observation_diagnostics
cells
exclusions
model_summary
advisories
provenance
```

### `MarginalMeansResult`

```text
means
contrasts
exclusions
provenance
```

### `MixedEffectsResult`

```text
status
reason
fixed_effects
variance_components
random_effects
exclusions
model_summary
advisories
provenance
```

### `RepresentationComparisonResult`

```text
cca
cka
procrustes
distance_similarity
mantel
exclusions
alignment
provenance
```

### `CCAResult`

```text
status
reason
correlations
x_weights
y_weights
x_loadings
y_loadings
x_scores
y_scores
advisories
provenance
```

### `CompositionalResult`

```text
transformed
variation_matrix
aitchison_distances
zero_replacement
provenance
```

### `BayesianEDAResult`

```text
means
mean_differences
provenance
```

### `AnomalyResult`

```text
scores
methods
exclusions
provenance
```

## Unified result graph

`UnifiedAnalysisResult` has one optional field per top-level `AnalysisBlock`, plus:

- `feature_alignment`;
- `executed_blocks`;
- unified provenance.

Only enabled blocks are expected to contain results.

Use:

```python
result.component("pca")
```

or:

```python
result.components
```

to access components programmatically. `components` is an immutable mapping containing only executed blocks.

`result.summary()` returns a lightweight serialization-safe overview suitable for CLI/application metadata.

## Why results are tables instead of hidden model objects

Most scientific outputs are exposed as Polars DataFrames (`pl.DataFrame`) rather than requiring callers to understand internal SciPy/statsmodels/sklearn result classes. This gives downstream tools a stable, tabular contract for:

- visualization;
- export;
- filtering;
- comparison;
- report generation;
- local application rendering.

Ruddy may use external scientific libraries internally, but its public result contract is intentionally its own.
