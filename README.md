# Ruddy

Ruddy is a domain-agnostic Python library for statistical exploratory data analysis of tabular datasets and high-dimensional feature representations.

## Scientific scope

The core library is intended to provide:

- dataset profiling and missingness analysis;
- univariate analysis;
- bivariate associations and group comparisons;
- effect sizes and multiple-testing correction;
- statistical outlier diagnostics;
- multivariate analysis;
- factorial analysis;
- PCA and exploratory manifold projections;
- explicit result, status, diagnostic, and provenance contracts.

## Command-line interface

Ruddy includes a first-class `ruddy` CLI entry point from the project skeleton.
Scientific commands are added as their corresponding library capabilities are implemented.

```bash
ruddy --help
ruddy --version
```

## Boundary

Ruddy does **not** interpret the scientific meaning of observations, generate molecular or biological representations, create plots, or generate reports. Visualization, reporting, and domain-specific interpretation belong to downstream applications such as Ruddy Web.

This repository currently contains the initial project skeleton only. Scientific implementation is added phase by phase.
