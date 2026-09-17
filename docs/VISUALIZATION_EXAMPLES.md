# Visualization examples

Visualization is intentionally external to the Ruddy scientific core. The marimo examples under `examples/` demonstrate how the structured outputs can support publication-oriented and interactive figures.

The plotting layer uses example-only dependencies (the `examples` dependency group) such as Matplotlib and Plotly. No Matplotlib/Plotly imports are required by `ruddy`.

## Catalog

### `01_profiling_univariate.py`

Demonstrates:

- dataset and missingness profiling;
- overall numerical distributions;
- distributions stratified by group/class;
- KDE/density views;
- ECDF;
- violin/box/raw-observation combinations;
- categorical frequencies.

This example shows that Ruddy's descriptive tables are sufficient to build more than simple histograms.

### `02_bivariate_dependence.py`

Demonstrates:

- strong scatterplots by group;
- group-specific tendencies/centroids;
- linear versus nonlinear relationships;
- Pearson/Spearman/general-dependence comparison;
- distance correlation;
- mutual information;
- partial correlations;
- contingency residual heatmaps;
- interactive scatter exploration.

### `03_groups_posthoc_intervals.py`

Demonstrates:

- response distributions by group;
- violin/box/raw observations;
- group confidence intervals;
- Tukey–Kramer comparisons;
- Games–Howell comparisons;
- pairwise difference/effect-size forests;
- matrix summaries of pairwise effects.

### `04_factorial_marginal_mixed.py`

Demonstrates:

- factorial main effects and interactions;
- effect-size summaries;
- raw and model-adjusted interaction plots;
- estimated marginal means;
- EMM contrasts;
- residual/model diagnostics;
- Cook's distance/leverage/influence;
- mixed-effect fixed/random components.

### `05_feature_spaces_projections.py`

Demonstrates:

- PCA scree/cumulative-variance plots;
- PCA scores colored by factor or continuous response;
- confidence-style ellipses for visualization;
- feature loadings;
- biplot-style views;
- t-SNE exploratory projection;
- interactive PCA exploration.

### `06_multivariate_permanova.py`

Demonstrates:

- covariance/correlation heatmaps;
- VIF and condition diagnostics;
- Mahalanobis ranking/threshold views;
- projected feature space by group;
- PERMANOVA separation;
- PERMDISP distance-to-centroid distributions.

### `07_representation_comparison.py`

Demonstrates:

- CKA similarity views;
- canonical-correlation spectrum;
- CCA score relationships;
- feature/canonical loadings;
- pairwise-distance-space comparison;
- Mantel results;
- Procrustes geometry;
- interactive canonical-space plots.

This is one of the key examples for the numerical-representation use case.

### `08_compositional.py`

Demonstrates:

- composition profiles;
- explicit zero replacement;
- CLR/ILR transformed spaces;
- variation matrix;
- Aitchison distance structure;
- within-versus-between group compositional geometry;
- PCA on ILR coordinates.

### `09_outliers_anomaly.py`

Demonstrates different notions of unusualness:

- IQR;
- modified robust Z;
- classical/robust Mahalanobis;
- Isolation Forest;
- LOF;
- score distributions;
- method-agreement views;
- anomalies overlaid on a projection.

### `10_bayesian_eda.py`

Demonstrates:

- posterior means;
- credible intervals;
- posterior group differences;
- probability of direction;
- ROPE probability;
- density and interval visualizations.

### `11_numerical_representation_framework.py`

The central representation-analysis demo. It compares multiple aligned numerical spaces through:

- PCA views;
- group separation;
- PERMANOVA/PERMDISP;
- CKA;
- CCA;
- distance similarity;
- Mantel;
- anomaly behavior.

The example is deliberately agnostic about the source of each representation; each is simply a `FeatureMatrix` with shared observation identity.

### `12_end_to_end_generic_eda.py`

Shows a complete domain-agnostic workflow from tabular loading/roles through unified analysis and feature-space integration.

### `13_visualization_gallery.py`

Acts as a visual catalog for future local-application design. It maps result types to candidate visual forms and provides a practical starting point for the later product specification.

## Visualization design principle

Every figure should be traceable to a Ruddy result object/table. The downstream visualization layer should not recompute the underlying statistics.

Examples:

```text
FactorialResult.effects
    → effect-size plot

MarginalMeansResult.means
    → adjusted mean/interaction plot

PermutationGroupResult.distances_to_centroid
    → PERMDISP distribution plot

RepresentationComparisonResult.cka
    → CKA heatmap/matrix

CCAResult.x_scores + y_scores
    → canonical-score comparison

AnomalyResult.scores
    → method score/ranking/agreement plot
```

This separation should remain a core requirement for the future local visual application.
