# Changelog

All notable changes to Ruddy are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## 0.2.0

- `--config PATH` reads option values from a YAML/JSON file (defaults < file < command-line flags) in `pipeline` and every `inspect`, `analyze`, `model` and `project` subcommand. Path options stay on the command line.
- `ruddy validate-config [COMMAND...] CONFIG [--json]` (e.g. `validate-config project pca cfg.yaml`) checks a config without reading data (the pipeline's by default).
- `ruddy.validate_config(config, command="pipeline")` and `ruddy.ConfigIssue` expose the same check in Python (mapping or file path; returns a list of issues, never raises for config problems).
- Every command that writes to `--output-dir` also writes `DIR.provenance.json`, following the shared ecosystem CLI contract. `pyyaml` is now a runtime dependency.

### Breaking changes

- Exit codes follow the shared ecosystem CLI contract: data and runtime errors (`RuddyError`, `OSError`, `ValueError`) now exit `3` instead of `2`, and unexpected errors exit `4` with a traceback instead of `1`. Usage and config errors still exit `2`.
- Per-block provenance files are named `<block>_provenance.json` after the block they describe: `profiling_provenance.json` (was `provenance.json`), `groups_provenance.json` (was `grouped_…`), `outliers_provenance.json` (was `outlier_…`), `intervals_provenance.json` (was `interval_…`), `permanova_provenance.json` (was `permutation_group_…`), `mixed_effects_provenance.json` (was `mixed_…`), and `tsne_provenance.json` / `umap_provenance.json` (was `projection_provenance.json`).
- Per-record output tables name their identifier column `id` and put it first (was `observation_id`), e.g. `pca_scores.csv` starts `id,source_row_index,PC1,...`.
- A table or matrix without an id column still gets generated positional ids, but the CLI now warns on stderr naming the option to set and records `extensions.generated_ids: true` in the provenance sidecar.

## 0.1.0

First public release.

- `TabularDataset` and `FeatureMatrix` data contracts, aligned by observation ID.
- Profiling, univariate, bivariate, group, post hoc, factorial, mixed-effects,
  multivariate, compositional, anomaly, Bayesian and representation-comparison
  analyses, plus PCA, t-SNE and optional UMAP projections.
- A unified `analyze` entry point driven by `AnalysisConfig`.
- The `ruddy` command-line interface.
- Polars result tables with explicit exclusions, degenerate and skipped states,
  and structured provenance.
