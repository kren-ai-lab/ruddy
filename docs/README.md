# Ruddy technical documentation

This directory documents the scientific MVP as implemented in the current codebase. It is intended to answer four practical questions:

1. **What is implemented?**
2. **What mathematical/statistical policy does each method follow?**
3. **What structured outputs are produced?**
4. **What does Ruddy deliberately refuse to infer or do automatically?**

The documentation is code-oriented and describes the current public behavior, not a future product specification.

## Recommended reading order

| Document | Purpose |
| --- | --- |
| [`SCIENTIFIC_SCOPE.md`](SCIENTIFIC_SCOPE.md) | Scope, boundaries, scientific design and intended role |
| [`DATA_CONTRACTS.md`](DATA_CONTRACTS.md) | `TabularDataset`, `FeatureMatrix`, roles, kinds, alignment, annotations |
| [`METHOD_INVENTORY.md`](METHOD_INVENTORY.md) | Fast capability matrix of everything currently implemented |
| [`METHODS.md`](METHODS.md) | Detailed scientific/methodological reference |
| [`STATISTICAL_POLICIES.md`](STATISTICAL_POLICIES.md) | Cross-cutting policies: missingness, scaling, FDR, permutations, assumptions, degeneracies |
| [`RESULTS_AND_PROVENANCE.md`](RESULTS_AND_PROVENANCE.md) | Result objects, status/reason/advisory contracts, exclusions and provenance |
| [`OUTPUT_SCHEMAS.md`](OUTPUT_SCHEMAS.md) | Exact declared column schemas for tabular outputs |
| [`UNIFIED_ANALYSIS.md`](UNIFIED_ANALYSIS.md) | `AnalysisConfig`, `AnalysisBlock` and `ruddy.analyze()` |
| [`CLI_REFERENCE.md`](CLI_REFERENCE.md) | Current functional CLI and command-to-analysis mapping |
| [`VISUALIZATION_EXAMPLES.md`](VISUALIZATION_EXAMPLES.md) | External visualization examples and what each one demonstrates |
| [`TESTING_AND_REPRODUCIBILITY.md`](TESTING_AND_REPRODUCIBILITY.md) | Test layout, robustness suite, deterministic stochastic analyses and feature freeze |
| [`PUBLIC_API.md`](PUBLIC_API.md) | Generated inventory of current public Python call signatures |

## Current scientific layers

```text
TabularDataset                         FeatureMatrix
     │                                      │
     ├─ profiling                           ├─ preprocessing
     ├─ univariate                          ├─ PCA / t-SNE / UMAP
     ├─ diagnostics                         ├─ covariance / VIF
     ├─ bivariate                           ├─ Mahalanobis
     ├─ dependence                          ├─ anomaly detection
     ├─ contingency                         └─ compositional analysis
     ├─ groups                                  │
     ├─ post-hoc                                │
     ├─ factorial                               │
     ├─ marginal means                          │
     ├─ mixed effects                           │
     └─ Bayesian EDA                            │
            │                                   │
            └──────── PERMANOVA/PERMDISP ──────┘
                                                │
FeatureMatrix A ───── representation comparison ───── FeatureMatrix B
                   CCA / CKA / Procrustes /
                   distance similarity / Mantel
```

## Core versus visualization

The current scientific package under `ruddy/` produces data structures and numerical results only. Visualization examples are intentionally kept under `examples/`. This boundary protects the statistical core from plotting dependencies while giving downstream interfaces a clear structured contract to consume.
