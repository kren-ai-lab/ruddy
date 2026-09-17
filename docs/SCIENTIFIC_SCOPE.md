# Scientific scope

## Purpose

Ruddy is a statistical exploratory analysis engine for two closely related data forms:

1. **tabular datasets** in which rows are observations and columns have explicit statistical roles and observed data kinds;
2. **numerical feature spaces** in which the same observations are represented by vectors, embeddings, descriptors or other numeric encodings.

The library is domain-agnostic. Ruddy does not need to know whether an observation is a protein, peptide, molecule, structure, experiment, sample or any other scientific object. Domain interpretation remains outside the statistical engine.

## Intended scientific role

Ruddy is designed to support the exploratory and comparative analysis of numerical representations. In the broader Numerical Representation Framework, other components may produce sequence-derived, structural or tabular descriptor representations; Ruddy receives those outputs and provides the statistical/EDA layer required to characterize and compare them.

The same API remains useful for ordinary tabular EDA without any representation-framework context.

## What belongs to the scientific core

The current scientific MVP includes:

- data validation and identity contracts;
- variable typing and statistical roles;
- missingness and quality profiling;
- descriptive statistics;
- distribution and dispersion diagnostics;
- linear, rank-based and general dependence analysis;
- mixed-type bivariate inference;
- effect sizes and confidence intervals;
- multiple-testing correction;
- group-wise response analysis;
- post-hoc comparisons;
- outlier diagnostics;
- PCA and exploratory manifold projections;
- covariance, collinearity and multivariate distance diagnostics;
- MANOVA;
- PERMANOVA and PERMDISP;
- factorial ANOVA/ANCOVA and interactions;
- estimated marginal means and contrasts;
- scoped mixed-effects modeling;
- pairwise numerical-representation comparison;
- compositional-data transformations and geometry;
- scoped Bayesian exploratory estimation;
- multivariate anomaly diagnostics;
- unified orchestration and CLI access.

## What does not belong to the current scientific core

Ruddy currently does not aim to be a general-purpose modeling environment. The following are outside the scientific MVP:

- supervised predictive modeling;
- clustering workflows;
- survival analysis;
- general time-series modeling;
- repeated-measures-specific frameworks;
- arbitrary generalized linear models;
- quantile regression;
- unrestricted Bayesian model specification;
- arbitrary hierarchical Bayesian models;
- PLS;
- domain-specific feature generation;
- automatic biological or chemical interpretation;
- automatic report generation;
- plotting inside `ruddy`.

Some of these may be future extensions, but they are not implicit promises of the current API.

## Scientific automation boundary

Ruddy deliberately avoids many forms of automatic statistical decision-making.

A diagnostic result is **information**, not a command to mutate the analysis plan. For example:

- a failed normality diagnostic does not automatically replace a requested parametric test;
- heteroscedasticity does not automatically activate robust covariance or Games–Howell;
- a high-dimensional matrix is not silently reduced before CCA, MANOVA or Mahalanobis analysis;
- a singular covariance matrix is not silently repaired by pseudo-inverse;
- zeros in compositional data are not automatically replaced;
- outliers/anomalies are not automatically removed;
- sparse matrices are not silently densified for methods that require dense arrays.

The library favors explicit refusal or an observable `degenerate`/`skipped` state over hidden transformations that make a method appear to work.

## Interpretive boundary

Ruddy returns statistical quantities and structured diagnostics. It does not claim that:

- association implies causation;
- statistical significance implies scientific relevance;
- anomaly scores identify erroneous observations;
- nonlinear projection coordinates are directly inferential variables;
- group separation in a representation implies a superior predictive model;
- similarity between representation spaces implies identical biological information.

Those interpretations require scientific context and are deliberately left to the user or downstream application.

## Visualization boundary

Visualization is a first-class downstream use case but not currently implemented in the core package. The notebooks under `examples/notebooks/` demonstrate how structured Ruddy results can be rendered using external plotting libraries.

This boundary is intentional:

```text
scientific input
    ↓
Ruddy core
    ↓
structured results + provenance
    ↓
notebooks / future local visual application
    ↓
plots, interactive exploration, figure export
```

The future local visual application should consume the same result contracts rather than reimplementing statistical logic.
