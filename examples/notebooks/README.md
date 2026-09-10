# Ruddy strong analytical and visualization notebooks

Phase 11 is the demonstration layer for Ruddy. These notebooks intentionally go beyond smoke-test plots: each notebook combines structured Ruddy results with raw observations where appropriate to show distributions, group differences, uncertainty, nonlinear dependence, multivariate geometry, representation similarity and anomaly structure.

Visualization dependencies remain entirely outside `src/ruddy`. The scientific core produces structured results; the notebooks demonstrate how a future local visual application can render those results.

## Notebook catalog

| Notebook | Analytical story | Visual demonstrations |
|---|---|---|
| `01_profiling_univariate.ipynb` | quality + distributions | missing/non-finite overview, violin+box+raw points, density, ECDF, faceted histograms, source distributions |
| `02_bivariate_dependence.ipynb` | linear vs nonlinear dependence | grouped scatter/trends, Pearson and distance-correlation heatmaps, dependence comparison, partial correlation ranking, contingency residual heatmap, interactive scatter |
| `03_groups_posthoc_intervals.ipynb` | magnitude + uncertainty across groups | violin/density views, group summaries, Tukey, Games–Howell, effect-size CIs, pairwise difference matrix |
| `04_factorial_marginal_mixed.ipynb` | effects + interactions + hierarchy | partial-eta² ranking, raw interaction, adjusted EMM interaction, EMM contrasts, residual/influence diagnostics, random effects |
| `05_feature_spaces_projections.ipynb` | high-dimensional organization | scree/cumulative variance, multiple PCA score views, ellipses, loading ranking, biplot, t-SNE, interactive PCA |
| `06_multivariate_permanova.ipynb` | geometry + group separation/dispersion | covariance/correlation structure, VIF, Mahalanobis ranking, PCA group view, PERMDISP distributions, group dispersion |
| `07_representation_comparison.ipynb` | compare aligned numerical spaces | CKA matrix, CCA spectrum/scores, distance-space scatter, Mantel, Procrustes segments, pairwise metric profiles, interactive CCA |
| `08_compositional.ipynb` | simplex-aware EDA | stacked compositions, zero diagnostics, variation matrix, CLR structure, ILR-PCA, Aitchison within/between comparison |
| `09_outliers_anomaly.ipynb` | competing notions of extremeness | univariate flags, anomaly-score distributions, method agreement, score comparison, PCA overlay, classical vs robust Mahalanobis |
| `10_bayesian_eda.ipynb` | posterior uncertainty | credible-interval forests, posterior group differences, direction vs ROPE, sign probabilities |
| `11_numerical_representation_framework.ipynb` | three aligned representation spaces | parallel PCA, PERMANOVA R² comparison, CKA matrix, geometry similarity profile, anomaly burden, interactive aligned-space explorer |
| `12_end_to_end_generic_eda.ipynb` | domain-agnostic full workflow | group distributions, summaries, PCA, statistical-vs-algorithmic flags, representation metrics, unified result map |
| `13_visualization_gallery.ipynb` | visual requirements gallery | representative distribution, relationship, group, factorial, projection, representation and anomaly plot families |

## Demo data

`examples/data/generate_demo_data.py` creates deterministic, domain-agnostic datasets with known structure, nonlinear dependence, group/source effects, aligned numerical representations, compositions and deliberate anomalies.

## Notebook-only dependencies

```bash
python -m pip install -r examples/notebooks/requirements-notebooks.txt
```

## Validation

Fast code-path validation without rendering notebook outputs:

```bash
python tools/run_notebook_gate.py --mode python
```

Full Jupyter execution for one or all notebooks can be performed with `nbconvert`; the checked-in notebooks are already executed and contain static figure outputs. Selected notebooks also contain interactive Plotly views.

The Phase-11 tests require multiple rendered figures per notebook and verify that visualization packages do not leak into the Ruddy core.
