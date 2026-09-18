# CLI reference

Use `ruddy` to inspect a dataset, run analyses, fit a model or project a
feature space from the terminal.

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
and `.npz` for SciPy sparse matrices. A `.txt` table is tab-separated; an
IDs file supplied with `--feature-ids-file` contains one ID per line.

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

Supply shared observation IDs when combining independently loaded files.

## Structured artifacts

The CLI writes result tables as CSV and summaries/provenance as JSON. Matrix
exports include a `feature` or `observation_id` column for row labels; no
implicit row index is written. CSV does not retain type metadata, so specify
ID dtypes when reading exports back for alignment.

The following table lists the main artifacts. Commands also write applicable
provenance, status, exclusion and advisory records. Files depend on the requested
blocks and options.

| Analysis | Main result files |
| --- | --- |
| Profiling | `overview.json`, `columns.csv`, `missingness.csv`, `pairwise_completeness.csv`, `missingness_patterns.csv` |
| Univariate | Profiling files plus `numeric_statistics.csv`, `categorical_statistics.csv`, `categorical_frequencies.csv`, `datetime_statistics.csv` |
| Outliers | `outlier_summaries.csv`, `numeric_quality.csv`, `outlier_flags.csv` (when requested) |
| Bivariate | `correlations.csv`, `comparisons.csv`, `categorical_associations.csv` |
| Dependence | `partial_correlations.csv`, `distance_correlations.csv`, `mutual_information.csv` |
| Contingency | `contingency_summary.csv`, `contingency_cells.csv` |
| Intervals | `mean_intervals.csv`, `correlation_intervals.csv`, `mean_difference_intervals.csv`, `effect_size_intervals.csv`, `odds_ratio_intervals.csv` |
| Factorial | `factorial_effects.csv`, `factorial_coefficients.csv`, `factorial_cells.csv`, `factorial_diagnostics.csv`, `factorial_model_summary.json` |
| Marginal means | `marginal_means.csv`, `marginal_contrasts.csv` |
| Mixed effects | `mixed_fixed_effects.csv`, `mixed_variance_components.csv`, `mixed_random_effects.csv`, `mixed_model_summary.json` |
| PCA | `pca_scores.csv`, `pca_loadings.csv`, `pca_variance.csv` |
| t-SNE/UMAP | `projection_coordinates.csv`, `projection_exclusions.csv` |
| Covariance | `covariance_matrix.csv`, `pearson_matrix.csv`, `spearman_matrix.csv`, `pairwise_counts.csv` |
| Collinearity | `collinearity.csv`, `collinearity_condition_spectrum.csv` |
| Mahalanobis | `mahalanobis_methods.csv`, `mahalanobis_distances.csv` |
| MANOVA | `manova_tests.csv`, `manova_factor_levels.csv`, `manova_model_summary.json` |
| PERMANOVA/PERMDISP | `permutation_group_summary.csv`, `permutation_group_levels.csv`, `permdisp_distances.csv` |
| Representation comparison | `cca_correlations.csv`, CCA score/weight/loading tables, `cka.csv`, `procrustes.csv`, `distance_similarity.csv`, `mantel.csv` |
| Composition | `compositional_transformed.csv`, `variation_matrix.csv`, `aitchison_distances.csv`, `zero_replacement.csv` |
| Bayesian EDA | `bayesian_means.csv`, `bayesian_mean_differences.csv` |
| Anomalies | `anomaly_scores.csv`, `anomaly_methods.csv` |

## Limitations

The CLI does not accept reusable configuration files or show progress during
long analyses. Use Python `AnalysisConfig` for reusable multi-analysis settings;
see [unified analysis](UNIFIED_ANALYSIS.md).
