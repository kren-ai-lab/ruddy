# Statistical policies

This document collects the cross-cutting rules that apply across Ruddy. These policies are as important as the individual tests because they define what the library will and will not do automatically.

## 1. No silent type coercion

`TabularDataset` does not reinterpret numeric-looking strings as numbers. Statistical kind inference follows the observed Polars dtype unless the caller explicitly provides a kind override.

This prevents accidental changes such as treating identifiers or coded categories as continuous measurements.

## 2. Roles and kinds are separate

A column can be numerically stored but statistically categorical when declared as a factor. Conversely, a numeric response remains numeric while gaining response semantics.

Dispatch therefore considers both:

- `ColumnKind` — what the observed data look like;
- `ColumnRole` — how the user intends Ruddy to use them.

## 3. Observation identity precedes row position

Whenever two scientific objects are combined, Ruddy prefers observation IDs over positional assumptions. This applies to:

- external annotations;
- tabular data + feature matrices;
- paired representation spaces;
- PERMANOVA metadata alignment.

Strict alignment requires exact coverage. Partial alignment remains visible through an `AlignmentReport`.

## 4. No silent row deletion

Individual analyses may require pairwise-finite, group-specific or complete-case data. Exclusions are handled locally to the analysis rather than mutating the source dataset.

Where row-level exclusion is relevant, Ruddy returns an exclusions table containing observation ID, source row index, stage and reason.

## 5. Missing and non-finite values are distinct

For numeric data:

- Polars `null` or float `NaN` is counted as missing (`n_missing`);
- `+inf`/`-inf` is present but non-finite (`n_non_finite`).

In result tables, missing values are represented exclusively as Polars `null`, never `NaN`. Most numerical statistics use finite values only while retaining separate counts of missing and non-finite observations.

## 6. No automatic imputation

Ruddy does not currently impute missing values as part of EDA. If an analysis requires complete observations, incomplete rows are excluded only for that method and the exclusion is reported.

## 7. No automatic scaling

Feature-space methods use explicit `ScalingMethod` values:

- `none`;
- `standard`;
- `robust`;
- `minmax`.

The default is `none` unless a specific method defines another documented default, such as standard scaling for CCA in representation comparison.

## 8. No silent sparse-to-dense conversion

Sparse matrices are supported only where the implementation can operate on them without hidden memory expansion. If a method currently needs dense input, Ruddy raises or skips explicitly.

The caller must choose when to densify a large representation.

## 9. No automatic dimensionality reduction

Ruddy never inserts PCA before a high-dimensional method simply to make it run.

Examples:

- singular/high-dimensional Mahalanobis is reported as not feasible;
- CCA enforces effective-rank limits;
- MANOVA limits response dimensionality and requires rank;
- Procrustes requires equal dimensions;
- sparse PCA is not obtained by hidden dense conversion.

If dimensionality reduction is scientifically appropriate, the user explicitly runs PCA and passes the resulting `FeatureMatrix` downstream.

## 10. No hidden pseudo-inverse or regularization

When a method depends on an invertible/full-rank design, Ruddy treats rank deficiency as a scientific condition rather than silently substituting a pseudo-inverse/regularized estimator.

This policy is particularly important for:

- covariance/Mahalanobis;
- VIF;
- CCA;
- MANOVA;
- factorial models.

## 11. Diagnostics never choose tests

Normality, heteroscedasticity and other assumptions are returned as diagnostics. They do not alter methods requested elsewhere.

Ruddy therefore does **not** implement rules such as:

```text
Shapiro p < 0.05 → replace t-test with Mann–Whitney
Breusch–Pagan p < 0.05 → automatically use HC3
Levene p < 0.05 → automatically choose Games–Howell
```

The caller can use the diagnostics to make that decision explicitly.

## 12. Parametric/non-parametric alternatives are explicit

Bivariate group comparison offers both parametric and rank-based methods, but users select the methods to calculate. The existence of both outputs does not make one the automatic fallback for the other.

## 13. Post-hoc choice is explicit

Tukey–Kramer and Games–Howell can both be requested. Ruddy does not infer the correct post-hoc family from an earlier test.

## 14. Factorial SS type is explicit

Type II versus Type III sums of squares are user choices. Balance, interactions or diagnostics do not change `ss_type` automatically.

Factorial models use sum-to-zero contrasts so Type III effects are not defined relative to an arbitrary treatment-reference category.

## 15. Robust covariance is explicit

HC0/HC1/HC2/HC3 covariance is available for factorial models only when the caller requests it. A heteroscedasticity diagnostic does not toggle robust covariance.

## 16. Multiple-testing families are explicit

Ruddy supports:

- no correction;
- Benjamini–Hochberg FDR.

Correction is applied independently within a `family_id`. The output also records `family_size` and the correction method.

Only rows with:

```text
status == "ok"
AND finite p_value
```

belong to the inferential family.

This is important for statistic-only runs such as permutation analyses with zero permutations: the statistic remains `ok`, but there is no p/q value to correct.

## 17. Effect sizes accompany inference where implemented

Ruddy separates magnitude from significance and provides effect sizes for major group/association analyses:

- Hedges' g with Welch t-test;
- Cliff's delta with Mann–Whitney;
- eta squared with Welch ANOVA;
- epsilon squared with Kruskal–Wallis;
- bias-corrected Cramér's V with chi-square;
- eta/partial eta/omega/partial omega squared in factorial models.

Effect sizes can themselves be undefined in degenerate data and are not fabricated.

## 18. Confidence intervals are estimand-specific

Ruddy does not use one generic asymptotic interval everywhere.

Examples:

- mean → Student-t;
- Pearson correlation → Fisher-z;
- mean difference → Welch;
- rank correlation/effect size → bootstrap where implemented;
- odds ratio → log-Wald for strictly positive 2×2 cells.

Zero-cell odds-ratio intervals are not silently corrected.

## 19. Permutation inference is deterministic under a seed

Permutation-based methods use an explicit random state, including:

- distance correlation;
- mutual-information inference;
- PERMANOVA;
- PERMDISP;
- Mantel.

Finite permutation p-values use an add-one style correction where implemented, avoiding exact zero p-values from a finite randomization sample.

## 20. Zero permutations mean descriptive-only inference

Where supported, `n_permutations=0` returns the observed statistic while p/q values remain missing. This is not considered an analysis failure.

## 21. Bootstrap inference is deterministic under a seed

The generic bootstrap API exposes `random_state`. Current supported interval methods are percentile, basic and BCa. The full bootstrap distribution is not retained by default.

## 22. Nonlinear projection axes are exploratory

`t-SNE` and `UMAP` result objects explicitly contain:

```text
exploratory = True
inferential_allowed = False
```

This prevents downstream tooling from silently treating axes such as `UMAP1` as equivalent to PCA components.

## 23. PCA is an explicit linear derived representation

PCA scores/loadings/variance are returned with provenance, and principal components can be converted to a new `FeatureMatrix`. Downstream inference is still an explicit user choice, but Ruddy does not prohibit it structurally.

## 24. PERMANOVA is paired with PERMDISP

Ruddy intentionally runs both on the same distance-space grouping analysis. A PERMANOVA result should be interpreted alongside dispersion behavior rather than in isolation.

## 25. CCA has effective-rank guards

Canonical component count cannot exceed the effective rank supported simultaneously by the two spaces and sample size. Ruddy does not regularize classical CCA silently.

## 26. Procrustes dimensionality must match

Ruddy does not reduce or pad representations simply to make Procrustes possible. Unequal feature dimensionality returns a skipped result.

## 27. Compositional zeros are never replaced silently

Log-ratio methods require positive components. If zeros are present, the caller must explicitly request multiplicative zero replacement. The replacement operation is recorded.

## 28. Bayesian scope is explicit

The Bayesian block is a conjugate Normal–Inverse-Gamma exploratory estimator for means and binary-group mean differences. It is not an automatic Bayesian alternative to every frequentist analysis and does not expose an unrestricted model language.

## 29. Outliers and anomalies are diagnostics, not deletion policies

IQR, modified Z, Mahalanobis, robust Mahalanobis, Isolation Forest and LOF can flag observations. Ruddy never removes, winsorizes or edits those observations automatically.

Different methods embody different definitions of unusualness; disagreement between flags is scientifically possible and should remain visible.

## 30. Scientific status is preferable to generic NaN

When a requested result cannot be interpreted, Ruddy attempts to provide an explicit:

- `status`;
- `reason`;
- optional advisory.

A missing numerical field therefore has contextual metadata explaining why it is missing whenever the result contract supports that behavior.

## 31. Bias-corrected univariate skewness and kurtosis

Univariate skewness and kurtosis use SciPy bias-corrected estimators matching the adjusted Fisher–Pearson $G_1$ and unbiased excess $G_2$ definitions:
- Skewness: `scipy.stats.skew(values, bias=False)` (requires $n \ge 3$ finite observations and non-constant data).
- Kurtosis: `scipy.stats.kurtosis(values, bias=False)` (requires $n \ge 4$ finite observations and non-constant data).

## 32. Spearman correlation via average ranks

Spearman rank correlation is computed from complete-case observations using average ranks for tied values (`scipy.stats.rankdata(matrix, axis=0)` with default `method="average"`), followed by Pearson correlation on the resulting ranks.
