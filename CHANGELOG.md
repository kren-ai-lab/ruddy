# Changelog

All notable changes to Ruddy are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

## 0.2.0

- `ruddy pipeline --config PATH` reads option values from a YAML/JSON file (defaults < file < command-line flags). Path options stay on the command line.
- `ruddy validate CONFIG [--json]` checks a pipeline config without reading data.
- `ruddy pipeline --output-dir DIR` also writes `DIR.provenance.json`, following the shared ecosystem CLI contract. `pyyaml` is now a runtime dependency.

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
