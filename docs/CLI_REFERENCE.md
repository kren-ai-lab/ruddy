# CLI reference

Ruddy exposes a functional command-line interface through the `ruddy` entry point. The current CLI is intentionally scientific-first and visually basic; a polished CLI belongs to later productization work.

## Top-level commands

| Command | Purpose |
| --- | --- |
| `analyze` | Run the explicitly configured unified pipeline |
| `profile` | Profile a tabular dataset |
| `univariate` | Profiling + univariate statistics |
| `diagnostics` | Normality and grouped dispersion diagnostics |
| `bivariate` | Mixed-type bivariate associations and comparisons |
| `dependence` | Partial correlations, distance correlation, mutual information |
| `contingency` | Cell-level contingency diagnostics |
| `intervals` | Confidence intervals for common estimands |
| `groups` | Response-centric grouped analysis |
| `outliers` | IQR/robust-Z outlier and numeric-quality diagnostics |
| `posthoc` | Tukey–Kramer and Games–Howell |
| `factorial` | Factorial ANOVA/ANCOVA |
| `marginal-means` | Estimated marginal means and contrasts |
| `mixed-effects` | Scoped random-intercept/random-slope mixed model |
| `project` | PCA, t-SNE or UMAP |
| `multivariate` | Covariance, VIF/condition, Mahalanobis |
| `manova` | Main-effects MANOVA |
| `permanova` | PERMANOVA + PERMDISP |
| `representation` | CCA, CKA, Procrustes, distance similarity, Mantel |
| `compositional` | Log-ratio compositional analysis |
| `bayesian` | Scoped Bayesian exploratory estimation |
| `anomaly` | Isolation Forest and/or LOF |

Run `ruddy <command> --help` for the exact options supported by the installed version.

## Basic examples

```bash
ruddy profile data.csv \
  --id-column id \
  --output-dir results/profile
```

```bash
ruddy univariate data.csv \
  --id-column id \
  --response activity \
  --factor family \
  --exclude sequence \
  --output-dir results/univariate
```

```bash
ruddy dependence data.csv \
  --id-column id \
  --partial-covariate length \
  --dependence-permutations 999 \
  --output-dir results/dependence
```

```bash
ruddy factorial data.csv \
  --id-column id \
  --response activity \
  --factor family \
  --factor source \
  --covariate length \
  --interaction family:source \
  --ss-type 3 \
  --output-dir results/factorial
```

```bash
ruddy project embeddings.csv \
  --method pca \
  --id-column id \
  --n-components 20 \
  --scaling standard \
  --output-dir results/pca
```

```bash
ruddy permanova metadata.csv \
  --id-column id \
  --permanova-factor family \
  --feature-input embeddings.csv \
  --feature-id-column id \
  --permanova-metric cosine \
  --permanova-permutations 999 \
  --random-state 42 \
  --output-dir results/permanova
```

```bash
ruddy representation space_a.csv space_b.csv \
  --x-id-column id \
  --y-id-column id \
  --cca-components 5 \
  --cca-scaling standard \
  --distance-metric cosine \
  --mantel-permutations 999 \
  --random-state 42 \
  --output-dir results/representation
```

## Unified CLI

The `analyze` command exposes multiple `AnalysisBlock` values in one run. Blocks are enabled explicitly.

```bash
ruddy analyze data.csv \
  --id-column id \
  --response activity \
  --group family \
  --enable profiling \
  --enable univariate \
  --enable bivariate \
  --enable groups \
  --enable outliers \
  --output-dir results
```

Feature-space blocks can be added with a feature input:

```bash
ruddy analyze metadata.csv \
  --id-column id \
  --enable pca \
  --enable multivariate \
  --feature-input embeddings.npy \
  --feature-ids-file ids.txt \
  --projection-n-components 20 \
  --projection-scaling standard \
  --output-dir results
```

## Input identity

For CSV feature inputs, an ID column should be supplied when the matrix must align with a tabular dataset. For NPY matrices, a separate IDs file can be used. Representation comparison likewise accepts separate X/Y ID declarations.

The CLI should not be treated as permission to rely on row order when identities are scientifically meaningful.

## Structured artifacts

The CLI writes CSV and JSON files. Major artifact families currently include:

### Profiling/univariate

```text
overview.json
provenance.json
numeric_statistics.csv
categorical_statistics.csv
categorical_frequencies.csv
datetime_statistics.csv
numeric_quality.csv
```

### Bivariate/dependence

```text
correlations.csv
comparisons.csv
categorical_associations.csv
partial_correlations.csv
distance_correlations.csv
mutual_information.csv
contingency_summary.csv
contingency_cells.csv
```

### Intervals/groups

```text
mean_intervals.csv
correlation_intervals.csv
mean_difference_intervals.csv
effect_size_intervals.csv
odds_ratio_intervals.csv
```

Grouped analysis writes response/group summaries and comparison artifacts together with provenance.

### Factorial/model-based

```text
factorial_design_terms.csv
factorial_effects.csv
factorial_coefficients.csv
factorial_cells.csv
factorial_diagnostics.csv
factorial_observation_diagnostics.csv
factorial_exclusions.csv
factorial_model_summary.json
factorial_provenance.json

marginal_means.csv
marginal_contrasts.csv
marginal_exclusions.csv
marginal_means_provenance.json

mixed_fixed_effects.csv
mixed_variance_components.csv
mixed_random_effects.csv
mixed_exclusions.csv
mixed_model_summary.json
mixed_advisories.json
mixed_provenance.json
```

### Projection/multivariate

```text
pca_scores.csv
pca_loadings.csv
pca_variance.csv
projection_coordinates.csv
projection_exclusions.csv

covariance_matrix.csv
pearson_matrix.csv
spearman_matrix.csv
pairwise_counts.csv
covariance_feature_diagnostics.csv
covariance_condition_spectrum.csv
collinearity.csv
collinearity_condition_spectrum.csv
mahalanobis_methods.csv
mahalanobis_distances.csv
```

### MANOVA/PERMANOVA

```text
manova_tests.csv
manova_factor_levels.csv
manova_exclusions.csv
manova_model_summary.json

permutation_group_summary.csv
permutation_group_levels.csv
permdisp_distances.csv
permutation_group_exclusions.csv
permutation_group_alignment.json
permutation_group_advisories.json
```

### Representation comparison

```text
cca_correlations.csv
cca_x_weights.csv
cca_y_weights.csv
cca_x_loadings.csv
cca_y_loadings.csv
cca_x_scores.csv
cca_y_scores.csv
cka.csv
procrustes.csv
distance_similarity.csv
mantel.csv
representation_exclusions.csv
representation_alignment.json
representation_provenance.json
```

### Specialized

```text
compositional_transformed.csv
variation_matrix.csv
aitchison_distances.csv
zero_replacement.csv

bayesian_means.csv
bayesian_mean_differences.csv

anomaly_scores.csv
anomaly_methods.csv
anomaly_exclusions.csv
```

## Current limitation of the CLI layer

The CLI is feature-complete enough to expose the scientific core, but it is not yet the desired final user experience. Current productization gaps include:

- richer terminal formatting;
- clearer progress/report summaries;
- better grouped command hierarchy;
- reusable config files;
- visual feedback for warnings/degeneracies;
- improved artifact summaries.

Those items should be specified in the later handoff/product-design phase rather than being mixed into the scientific implementation.
