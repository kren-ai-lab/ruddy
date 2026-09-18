# CLI reference

Ruddy exposes its scientific core through the `ruddy` entry point, built on
Typer with Rich terminal output. Commands are grouped by intent: inspecting a
dataset, analyzing it, fitting an explicit model, or projecting a feature
space.

## Global options

| Option | Effect |
| --- | --- |
| `-h`, `--help` | Show help for the current command or group |
| `-v`, `--version` | Print `ruddy <version>` and exit |
| `-q`, `--quiet` | Suppress the Rich run summary |
| `--debug` | Re-raise errors with a full traceback instead of a clean message |

Global options are declared before the group, for example
`ruddy --quiet inspect profile data.csv`.

Every command ends with a run summary on **stderr**: the dataset or feature
matrix shape, the blocks that ran with `OK` or `WARN`, and where the artifacts
were written. Commands that print machine-readable JSON write it to **stdout**,
so that stream stays free of decoration and can be piped into `jq`.

A user-facing error is reported on stderr without a traceback and exits with
code `2`. Messages are actionable: an unknown column lists the available ones,
and an unsupported extension lists the supported ones. Invoking `ruddy` with no
command prints the help and exits with code `2`.

## Commands

### `ruddy pipeline` — the unified run

A single top-level command rather than a group: it orchestrates blocks from
every group, selected explicitly with `--enable`, under one configuration and
one provenance record.

### `ruddy inspect` — profiling and dataset diagnostics

| Command | Purpose |
| --- | --- |
| `profile` | Profile a tabular dataset |
| `univariate` | Profiling plus univariate statistics |
| `diagnostics` | Normality and grouped dispersion diagnostics |
| `outliers` | IQR/robust-Z outlier and numeric-quality diagnostics |
| `multivariate` | Covariance, VIF/condition, Mahalanobis |

### `ruddy analyze` — standalone analysis blocks

| Command | Purpose |
| --- | --- |
| `bivariate` | Mixed-type bivariate associations and comparisons |
| `dependence` | Partial correlations, distance correlation, mutual information |
| `contingency` | Cell-level contingency diagnostics |
| `intervals` | Confidence intervals for common estimands |
| `groups` | Response-centric grouped analysis |
| `posthoc` | Tukey-Kramer and Games-Howell |
| `representation` | CCA, CKA, Procrustes, distance similarity, Mantel |
| `compositional` | Log-ratio compositional analysis |
| `bayesian` | Scoped Bayesian exploratory estimation |
| `anomaly` | Isolation Forest and/or LOF |

### `ruddy model` — explicit inferential models

| Command | Purpose |
| --- | --- |
| `factorial` | Factorial ANOVA/ANCOVA |
| `marginal-means` | Estimated marginal means and contrasts |
| `mixed-effects` | Scoped random-intercept/random-slope mixed model |
| `manova` | Main-effects MANOVA |
| `permanova` | PERMANOVA + PERMDISP |

### `ruddy project` — feature-space projections

| Command | Purpose |
| --- | --- |
| `pca` | Principal component analysis |
| `tsne` | Exploratory t-SNE projection |
| `umap` | Exploratory UMAP projection |

Run `ruddy pipeline --help` or `ruddy <group> <command> --help` for the exact
options supported by the installed version.

## Accepted input formats

Tabular inputs are read by extension: `.csv`, `.tsv`, `.txt` as tab-separated,
and `.parquet`. Feature matrices additionally accept `.npy` for dense arrays
and `.npz` for SciPy sparse matrices. Every read and write goes through
`ruddy.core.io`, so the supported set is the same for every command.

## Basic examples

```bash
ruddy inspect profile data.csv \
  --id-column id \
  --output-dir results/profile
```

```bash
ruddy inspect univariate data.csv \
  --id-column id \
  --response activity \
  --factor family \
  --exclude sequence \
  --output-dir results/univariate
```

```bash
ruddy analyze dependence data.csv \
  --id-column id \
  --partial-covariate length \
  --dependence-permutations 999 \
  --output-dir results/dependence
```

```bash
ruddy model factorial data.csv \
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
ruddy project pca embeddings.csv \
  --id-column id \
  --n-components 20 \
  --scaling standard \
  --output-dir results/pca
```

```bash
ruddy model permanova metadata.csv \
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
ruddy analyze representation space_a.csv space_b.csv \
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

The `pipeline` command exposes multiple `AnalysisBlock` values in one run. Blocks are enabled explicitly.

```bash
ruddy pipeline data.csv \
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
ruddy pipeline metadata.csv \
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

The CLI writes CSV and JSON files via Polars without row indices (`ruddy.core.io.write_table`). Square and labeled matrices (such as `covariance_matrix.csv`, `pearson_matrix.csv`, `spearman_matrix.csv`, `pairwise_counts.csv`, `variation_matrix.csv`, `aitchison_distances.csv`, and projection tables like `pca_scores.csv` / `projection_coordinates.csv`) now include an explicit `feature` or `observation_id` first column preserving row labels and observation ID dtypes, instead of an unlabelled pandas index.

Major artifact families currently include:

### Profiling

```text
overview.json
provenance.json
columns.csv
missingness.csv
pairwise_completeness.csv
missingness_patterns.csv
```

### Univariate

Everything written by profiling, plus:

```text
univariate_provenance.json
numeric_statistics.csv
categorical_statistics.csv
categorical_frequencies.csv
datetime_statistics.csv
```

### Outliers

```text
outlier_summaries.csv
numeric_quality.csv
outlier_flags.csv          # only when flags are requested
outlier_provenance.json
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

## Current limitations of the CLI layer

The grouped hierarchy, the Rich run summary and the clean reporting of user
errors are in place. Remaining productization gaps include:

- reusable configuration files;
- progress reporting for long runs;
- richer artifact summaries beyond the output path.
