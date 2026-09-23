# Changelog

All notable changes to Ruddy are recorded here. Versions follow
[Semantic Versioning](https://semver.org/).

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
