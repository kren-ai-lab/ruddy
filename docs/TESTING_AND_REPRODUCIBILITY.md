# Testing, reproducibility and scientific freeze

Ruddy's current scientific MVP is protected by unit, integration, parity, pathological-data and notebook validation tests.

## Core validation commands

From a working checkout:

```bash
python -m compileall -q src tests
pytest -q
```

The reconstructed Phase 11 codebase used to prepare this documentation passes the complete test suite before documentation-only additions.

## Scientific freeze gate

Phase 10 introduced:

```bash
python tools/run_scientific_freeze_gate.py
```

The gate runs compilation, the dedicated Phase 10 torture suite and the global suite.

## What the torture suite targets

The scientific freeze is not based only on ordinary happy-path unit tests. It deliberately exercises conditions such as:

### Pathological tabular data

- all-missing columns;
- constant columns;
- non-finite numeric values;
- very small sample sizes;
- single-level factors;
- high-cardinality factors;
- extreme imbalance;
- duplicate/missing observation IDs;
- complete-case collapse;
- empty factorial cells;
- rank-deficient covariates/designs.

### High-dimensional feature spaces

- `n >> p`;
- `n ≈ p`;
- `p > n`;
- `p >> n`;
- low-rank spaces;
- perfect redundancy;
- sparse inputs.

The expected behavior is not always a successful statistic. In many cases the correct behavior is an explicit `degenerate`/`skipped` state or an exception preventing hidden densification/regularization.

### Representation invariants

Synthetic representation tests include:

- identical spaces;
- noisy copies;
- orthogonal rotations;
- global scaling;
- feature permutations;
- independent spaces;
- unequal feature dimensions;
- low-rank representations.

These are used to validate expected properties of CKA, CCA, Procrustes and distance-space comparison.

### Dependence patterns

Synthetic relationships include:

- linear dependence;
- monotonic nonlinear dependence;
- quadratic dependence;
- confounding;
- independent variables.

These cases ensure that Pearson/Spearman/partial correlation/distance correlation/mutual information behave distinctly and coherently.

### Compositional/anomaly behavior

Tests cover:

- zero-containing compositions;
- explicit zero replacement;
- closure/scale invariance;
- ILR/Aitchison geometry;
- extreme anomalies;
- anomaly-score reproducibility.

## Cross-surface parity

A critical test principle is:

```text
standalone Python API
        ↕
unified `analyze()` component
        ↕
CLI artifact
```

When two surfaces expose the same scientific operation, the result should be numerically equivalent within the defined tolerance rather than being implemented separately.

## Random-state policy

Stochastic methods expose explicit seeds through either function arguments or `AnalysisConfig.random_state`.

Current seeded methods include:

- t-SNE;
- UMAP (when installed);
- bootstrap intervals;
- permutation inference for dependence;
- PERMANOVA/PERMDISP;
- Mantel;
- Bayesian posterior sampling;
- Isolation Forest;
- robust covariance procedures where a random state is supported.

The same data, configuration and seed should reproduce the same output under the same dependency/runtime environment.

## Permutation p-values

Permutation analyses retain the requested permutation count in provenance/output. Statistic-only execution with zero permutations is supported in relevant APIs: the observed statistic remains available while p/q values are missing.

## Multiple-testing reproducibility

Benjamini–Hochberg correction uses deterministic stable sorting. `family_id`, `family_size` and `correction` are included in outputs so adjusted values can be audited.

## Notebook validation

Phase 11 adds tests under:

```text
tests/phase11/
```

These verify that the rich demo notebooks:

- exist and are executed;
- contain no error outputs;
- contain multiple rendered visualizations;
- use real Ruddy APIs/results;
- retain aligned demo observation IDs;
- keep plotting dependencies out of `src/ruddy`.

The notebooks can also be executed using the notebook gate tooling under `tools/`.

## Scientific freeze meaning

"Scientific freeze" means that the MVP method families and their core behavior are considered stable enough to stop adding features while documentation/product design proceeds. It does **not** mean that future versions can never add scientific methods.

After freeze, a change should be treated carefully if it alters:

- numerical definitions;
- default statistical policies;
- role/kind dispatch;
- hypothesis-family definitions;
- alignment semantics;
- scaling defaults;
- degeneracy behavior;
- result schemas;
- provenance semantics.

Such changes can invalidate notebooks, downstream applications or scientific reproducibility and should therefore receive dedicated tests and release notes.
