# Method inventory

This document is the compact capability map of the current Ruddy scientific MVP. For methodology, policies and edge cases, use [`METHODS.md`](METHODS.md) and [`STATISTICAL_POLICIES.md`](STATISTICAL_POLICIES.md).

## Data and profiling

| Capability | Public entry point | Input | Main output |
| --- | --- | --- | --- |
| Tabular data contract | `TabularDataset` | Polars / pandas DataFrame | immutable-by-contract tabular dataset |
| Feature-space contract | `FeatureMatrix` | NumPy / Polars / pandas / SciPy sparse | observation-aligned numeric matrix |
| External annotation alignment | `align_annotation_source()` | dataset + annotation table | `AlignedAnnotations` + coverage |
| Attach annotations | `attach_annotations()` | dataset + aligned annotations | new `TabularDataset` |
| Dataset profiling | `profile_dataset()` | `TabularDataset` | `ProfilingResult` |
| Column profiling | `profile_columns()` | `TabularDataset` | column-level Polars DataFrame |
| Missingness summary | `summarize_missingness()` | column profiles | Polars DataFrame |
| Missingness patterns | `summarize_missingness_patterns()` | `TabularDataset` | pattern table |
| Pairwise completeness | `pairwise_completeness()` | `TabularDataset` | pairwise completeness table |

## Univariate and distribution analysis

| Capability | Method/statistic | Entry point |
| --- | --- | --- |
| Numeric summaries | mean, SD, variance, range, median, quantiles, IQR, MAD, skewness, Fisher kurtosis, zeros | `analyze_univariate()` / `summarize_numeric_statistics()` |
| Categorical summaries | mode, counts, fractions, entropy, normalized entropy | `analyze_univariate()` / `summarize_categorical_statistics()` |
| Datetime summaries | min, max, range seconds | `analyze_univariate()` / `summarize_datetime_statistics()` |
| Numeric quality | missing/non-finite, unique finite, zero-IQR, zero-MAD states | `summarize_numeric_quality()` |
| Normality | Shapiro–Wilk | `analyze_distribution_diagnostics()` |
| Normality | D'Agostino K² | `analyze_distribution_diagnostics()` |
| Normality | Anderson–Darling normality test | `analyze_distribution_diagnostics()` |
| Grouped dispersion | Brown–Forsythe (median-centered Levene) | `analyze_distribution_diagnostics()` |
| Grouped dispersion | Fligner–Killeen | `analyze_distribution_diagnostics()` |

## Bivariate association and inference

| Variable relation | Method | Effect size / extra | Entry point |
| --- | --- | --- | --- |
| numeric ↔ numeric | Pearson | coefficient + p/q | `analyze_bivariate()` |
| numeric ↔ numeric | Spearman | coefficient + p/q | `analyze_bivariate()` |
| numeric ↔ numeric | Kendall | coefficient + p/q | `analyze_bivariate()` |
| numeric ↔ two groups | Welch t-test | Hedges' g | `analyze_bivariate()` |
| numeric ↔ two groups | Mann–Whitney U | Cliff's delta | `analyze_bivariate()` |
| numeric ↔ multiple groups | Welch ANOVA | η² | `analyze_bivariate()` |
| numeric ↔ multiple groups | Kruskal–Wallis | ε² | `analyze_bivariate()` |
| categorical ↔ categorical | Pearson χ² | bias-corrected Cramér's V | `analyze_bivariate()` |
| binary categorical ↔ binary categorical | Fisher exact | odds ratio | `analyze_bivariate()` |

## Extended dependence

| Capability | Method | Entry point |
| --- | --- | --- |
| Conditional dependence | partial Pearson correlation | `analyze_dependence()` |
| Conditional rank dependence | partial Spearman correlation | `analyze_dependence()` |
| General nonlinear dependence | distance correlation | `analyze_dependence()` |
| General dependence | mutual information | `analyze_dependence()` |
| General-dependence inference | permutation p-values | `analyze_dependence()` |

## Contingency diagnostics

| Capability | Output | Entry point |
| --- | --- | --- |
| Global association | χ², df, p/q, Cramér's V | `analyze_contingency_diagnostics()` |
| Expected-frequency diagnostics | minimum expected, counts/fraction <5 | `analyze_contingency_diagnostics()` |
| Cell diagnostics | expected count, Pearson residual, standardized residual | `analyze_contingency_diagnostics()` |
| χ² decomposition | per-cell contribution and contribution fraction | `analyze_contingency_diagnostics()` |
| Cell proportions | row and column fractions | `analyze_contingency_diagnostics()` |

## Confidence intervals and bootstrap

| Estimand | Interval strategy | Entry point |
| --- | --- | --- |
| Mean | Student-t | `mean_confidence_interval()` / `analyze_confidence_intervals()` |
| Pearson correlation | Fisher-z | `pearson_confidence_interval()` / `analyze_confidence_intervals()` |
| Spearman/Kendall correlation | bootstrap | `analyze_confidence_intervals()` |
| Mean difference | Welch | `welch_mean_difference_confidence_interval()` / `analyze_confidence_intervals()` |
| Hedges' g | bootstrap | `analyze_confidence_intervals()` |
| Odds ratio | log-Wald; no hidden zero-cell correction | `odds_ratio_confidence_interval()` / `analyze_confidence_intervals()` |
| Generic scalar statistic | percentile/basic/BCa bootstrap | `bootstrap_confidence_interval()` |

## Group-centric analysis

| Capability | Entry point |
| --- | --- |
| Response catalog | `summarize_response_catalog()` |
| Group coverage | `summarize_group_coverage()` |
| Numeric response summaries by group | `summarize_grouped_numeric_responses()` |
| Categorical response frequencies by group | `summarize_grouped_categorical_responses()` |
| Response-specific inferential comparisons | `analyze_groups()` |
| Annotation coverage | `summarize_annotation_coverage()` |

## Post-hoc analysis

| Method | Main policy | Entry point |
| --- | --- | --- |
| Tukey–Kramer HSD | pooled residual variance; unequal group sizes supported | `analyze_posthoc()` |
| Games–Howell | pair-specific variances and Welch–Satterthwaite df | `analyze_posthoc()` |

## Univariate outliers and quality

| Method | Definition | Entry point |
| --- | --- | --- |
| Tukey IQR | Q1 − k·IQR / Q3 + k·IQR, default k=1.5 | `analyze_outliers()` |
| Modified robust Z | 0.67448975·(x−median)/MAD, default threshold 3.5 | `analyze_outliers()` |
| Observation flags | optional; never deletes observations | `analyze_outliers(include_flags=True)` |

## Projections and preprocessing

| Capability | Policy | Entry point |
| --- | --- | --- |
| Explicit preprocessing | none / standard / robust / minmax | `prepare_features()` |
| PCA | scores, loadings, explained variance, cumulative variance, singular values | `analyze_pca()` |
| t-SNE | exploratory; fixed seed supported | `analyze_tsne()` |
| UMAP | exploratory; optional `umap-learn` dependency | `analyze_umap()` |
| Projection dispatch | explicit UMAP/t-SNE method | `analyze_projection()` |

## Multivariate diagnostics

| Capability | Method/output | Entry point |
| --- | --- | --- |
| Covariance structure | complete-case covariance | `analyze_covariance_structure()` |
| Correlation structure | Pearson + optional Spearman | `analyze_covariance_structure()` |
| Pairwise availability | finite observation counts | `analyze_covariance_structure()` |
| Matrix conditioning | rank, singular values, condition number/index | `analyze_covariance_structure()` |
| Collinearity | VIF and tolerance | `analyze_collinearity()` |
| Classical multivariate distance | Mahalanobis | `analyze_mahalanobis()` |
| Robust multivariate distance | MinCovDet Mahalanobis | `analyze_mahalanobis()` |
| Combined core | covariance + collinearity + Mahalanobis | `analyze_multivariate()` |

## Multivariate inference

| Capability | Method | Entry point |
| --- | --- | --- |
| Main-effects MANOVA | Wilks' lambda | `analyze_manova()` |
| Main-effects MANOVA | Pillai's trace | `analyze_manova()` |
| Main-effects MANOVA | Hotelling–Lawley trace | `analyze_manova()` |
| Main-effects MANOVA | Roy's greatest root | `analyze_manova()` |
| Group separation in distance space | PERMANOVA pseudo-F + R² | `analyze_permutation_group_structure()` |
| Group dispersion | PERMDISP | `analyze_permutation_group_structure()` |

## Factorial and model-based analysis

| Capability | Implemented behavior | Entry point |
| --- | --- | --- |
| Factorial ANOVA | main effects + interactions | `analyze_factorial()` |
| ANCOVA | numeric covariates | `analyze_factorial()` |
| Sum of squares | Type II and Type III | `analyze_factorial()` |
| Contrasts | sum-to-zero factor coding | `analyze_factorial()` |
| Robust covariance | HC0/HC1/HC2/HC3 optional | `analyze_factorial()` |
| Factorial effect sizes | η², partial η², ω², partial ω² | `analyze_factorial()` |
| Model coefficients | estimate, SE, t, p, CI | `analyze_factorial()` |
| Residual diagnostics | Shapiro, Jarque–Bera, Breusch–Pagan, Brown–Forsythe/Levene median, condition number | `analyze_factorial()` |
| Influence diagnostics | studentized residual, leverage, Cook's distance | `analyze_factorial()` |
| Estimated marginal means | equal weighting across nuisance-factor levels; covariates at observed means | `analyze_marginal_means()` |
| EMM contrasts | model-based pairwise contrasts + CI + optional FDR | `analyze_marginal_means()` |
| Mixed effects | random intercept + optional numeric random slopes | `analyze_mixed_effects()` |

## Representation-space analysis

| Capability | Purpose | Entry point |
| --- | --- | --- |
| Alignment | align two spaces by observation ID | `align_feature_matrices()` |
| CCA | shared canonical axes between two spaces | `analyze_representation_similarity()` |
| Linear CKA | representation similarity independent of equal feature count | `linear_cka()` / `analyze_representation_similarity()` |
| Procrustes | geometric alignment when dimensions match | `analyze_representation_similarity()` |
| Distance-space similarity | Pearson/Spearman correlation of pairwise distances | `analyze_representation_similarity()` |
| Mantel | permutation test for distance-space association | `analyze_representation_similarity()` |

## Compositional data analysis

| Capability | Entry point |
| --- | --- |
| Closure | `closure()` |
| Multiplicative zero replacement | `multiplicative_zero_replacement()` |
| Centered log-ratio | `clr_transform()` |
| Additive log-ratio | `alr_transform()` |
| Isometric log-ratio | `ilr_transform()` |
| Variation matrix | `variation_matrix()` |
| Aitchison distance | `aitchison_distance_matrix()` |
| Integrated compositional analysis | `analyze_composition()` |

## Bayesian exploratory analysis

| Capability | Entry point |
| --- | --- |
| Posterior mean for numeric variables | `analyze_bayesian_eda()` |
| Posterior mean difference for binary groups | `analyze_bayesian_eda()` |
| Credible interval | `analyze_bayesian_eda()` |
| Probability positive/negative | `analyze_bayesian_eda()` |
| Probability of direction | `analyze_bayesian_eda()` |
| ROPE probability | `analyze_bayesian_eda()` |

The current model is a scoped Normal–Inverse-Gamma conjugate analysis, not a general Bayesian modeling framework.

## Anomaly diagnostics

| Method | Output policy | Entry point |
| --- | --- | --- |
| Isolation Forest | anomaly score + model flag | `analyze_anomalies()` |
| Local Outlier Factor | anomaly score + model flag | `analyze_anomalies()` |

For both methods, larger Ruddy `anomaly_score` values correspond to more anomalous observations.

## Unified orchestration

`analyze()` can orchestrate the following explicit `AnalysisBlock` values:

```text
profiling
univariate
bivariate
groups
outliers
pca
tsne
umap
multivariate
manova
factorial
diagnostics
dependence
contingency
intervals
permanova
posthoc
marginal_means
mixed_effects
representation
compositional
bayesian
anomaly
```

Default blocks are only `profiling` and `univariate`.
