# Ruddy examples

These examples are the demonstration layer for Ruddy. They intentionally go beyond smoke-test plots: each one combines structured Ruddy results with raw observations where appropriate to show distributions, group differences, uncertainty, nonlinear dependence, multivariate geometry, representation similarity and anomaly structure.

Visualization dependencies remain entirely outside `ruddy`. The scientific core produces structured results; the examples demonstrate how a future local visual application can render those results.

## Catalog

| Example | Analytical story | Visual demonstrations |
|---|---|---|
| `01_profiling_univariate.py` | quality + distributions | missing/non-finite overview, violin+box+raw points, density, ECDF, faceted histograms, source distributions |
| `02_bivariate_dependence.py` | linear vs nonlinear dependence | grouped scatter/trends, Pearson and distance-correlation heatmaps, dependence comparison, partial correlation ranking, contingency residual heatmap, interactive scatter |
| `03_groups_posthoc_intervals.py` | magnitude + uncertainty across groups | violin/density views, group summaries, Tukey, Games–Howell, effect-size CIs, pairwise difference matrix |
| `04_factorial_marginal_mixed.py` | effects + interactions + hierarchy | partial-eta² ranking, raw interaction, adjusted EMM interaction, EMM contrasts, residual/influence diagnostics, random effects |
| `05_feature_spaces_projections.py` | high-dimensional organization | scree/cumulative variance, multiple PCA score views, ellipses, loading ranking, biplot, t-SNE, interactive PCA |
| `06_multivariate_permanova.py` | geometry + group separation/dispersion | covariance/correlation structure, VIF, Mahalanobis ranking, PCA group view, PERMDISP distributions, group dispersion |
| `07_representation_comparison.py` | compare aligned numerical spaces | CKA matrix, CCA spectrum/scores, distance-space scatter, Mantel, Procrustes segments, pairwise metric profiles, interactive CCA |
| `08_compositional.py` | simplex-aware EDA | stacked compositions, zero diagnostics, variation matrix, CLR structure, ILR-PCA, Aitchison within/between comparison |
| `09_outliers_anomaly.py` | competing notions of extremeness | univariate flags, anomaly-score distributions, method agreement, score comparison, PCA overlay, classical vs robust Mahalanobis |
| `10_bayesian_eda.py` | posterior uncertainty | credible-interval forests, posterior group differences, direction vs ROPE, sign probabilities |
| `11_numerical_representation_framework.py` | three aligned representation spaces | parallel PCA, PERMANOVA R² comparison, CKA matrix, geometry similarity profile, anomaly burden, interactive aligned-space explorer |
| `12_end_to_end_generic_eda.py` | domain-agnostic full workflow | group distributions, summaries, PCA, statistical-vs-algorithmic flags, representation metrics, unified result map |
| `13_visualization_gallery.py` | visual requirements gallery | representative distribution, relationship, group, factorial, projection, representation and anomaly plot families |

## Demo data

`examples/data/generate_demo_data.py` creates deterministic, domain-agnostic datasets with known structure, nonlinear dependence, group/source effects, aligned numerical representations, compositions and deliberate anomalies.

## Running

Each example is a [marimo](https://marimo.io) notebook stored as a plain Python file. Run it as a script, or open it interactively:

```bash
uv sync --group examples
MPLBACKEND=Agg uv run python examples/01_profiling_univariate.py   # script
uv run marimo edit examples/01_profiling_univariate.py             # notebook
```

`_helpers.py` holds the shared plotting functions; `run_ci_examples.sh` regenerates the demo data and runs every example, as CI does.
