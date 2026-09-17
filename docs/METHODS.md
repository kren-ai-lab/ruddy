# Detailed method reference

This document describes the scientific methods currently implemented in Ruddy. It focuses on **what is computed, which data are eligible, how edge cases are handled, and what each method returns**. It is not a tutorial on the statistical theory itself and it does not replace domain-specific scientific interpretation.

## 1. Dataset profiling

### 1.1 Dataset overview

`profile_dataset()` combines dataset-level and column-level profiling into a `ProfilingResult`.

The overview reports the basic table shape and role/kind composition. The detailed column table records:

- storage dtype;
- statistical role;
- observed data kind;
- whether the column is excluded or analysis-eligible;
- total, present and missing observations;
- missing fraction;
- number/fraction of unique values;
- finite and non-finite counts for numeric variables;
- all-missing and constant states.

Profiling does not transform the source table.

### 1.2 Missingness summary

`summarize_missingness()` converts the column profiles into a dedicated missingness table containing present/missing counts and fractions. Numeric columns additionally retain finite/non-finite counts.

A present `+inf` or `-inf` is **not** counted as missing; it is tracked as non-finite.

### 1.3 Pairwise completeness

`pairwise_completeness()` reports, for pairs of selected columns:

- total observations;
- jointly complete observations;
- incomplete observations;
- complete fraction.

The output is designed to show how much data would survive pairwise analyses before an inferential method is run.

### 1.4 Missingness patterns

`summarize_missingness_patterns()` counts row-level patterns of missing columns. Output is bounded by `max_patterns`, and truncation is explicitly recorded through the total number of patterns and unreported observation fraction.

This is descriptive only; Ruddy does not infer a missing-data mechanism such as MCAR/MAR/MNAR.

---

## 2. Univariate descriptive analysis

`analyze_univariate()` runs profiling plus numeric, categorical/boolean/factor and datetime summaries.

### 2.1 Numeric statistics

Eligible numeric columns exclude identifiers, excluded columns and factors. Numeric responses, variables and covariates are eligible.

Ruddy uses **finite observations only** for numerical statistics and separately records missing and non-finite values.

Implemented statistics include:

- `n_total`;
- `n_finite`;
- `n_missing`;
- `n_non_finite`;
- arithmetic mean;
- sample standard deviation (`ddof=1`);
- sample variance;
- minimum and maximum;
- range;
- configurable quantiles;
- median;
- interquartile range (IQR);
- unscaled median absolute deviation (MAD);
- skewness;
- Fisher-style kurtosis;
- zero count and zero fraction.

Important pathological states include:

- all missing → `skipped`;
- no finite numeric values → `skipped`;
- too few finite observations → `skipped`;
- constant variable → `degenerate`.

### 2.2 Categorical, boolean and factor statistics

Eligible categorical summaries include categorical columns, booleans and variables explicitly declared as factors.

Ruddy reports:

- present/missing counts;
- number of observed levels;
- mode, mode count and mode fraction;
- Shannon entropy using base 2;
- normalized entropy relative to `log2(n_levels)`;
- ranked frequency table;
- whether high-cardinality frequencies were truncated;
- unreported count/fraction when truncation occurs.

Ties in frequency ordering are resolved deterministically by level label.

### 2.3 Datetime summaries

Datetime analysis is deliberately descriptive rather than time-series-oriented. Ruddy reports:

- present/missing counts;
- earliest timestamp;
- latest timestamp;
- temporal range in seconds.

No trend, seasonality or temporal autocorrelation model is implied.

---

## 3. Distribution and dispersion diagnostics

`analyze_distribution_diagnostics()` provides standalone diagnostics. These results **never change another analysis automatically**.

### 3.1 Shapiro–Wilk

Applied to eligible numeric variables with at least three finite values. By default Ruddy does not run Shapiro–Wilk above 5000 observations; such rows are returned as `skipped` with `sample_size_exceeds_shapiro_limit`.

### 3.2 D'Agostino K²

Uses SciPy's normality test and requires at least eight finite observations.

### 3.3 Anderson–Darling normality diagnostic

Ruddy uses the statsmodels normal Anderson–Darling implementation and reports a statistic and p-value.

### 3.4 Brown–Forsythe

Grouped dispersion is assessed using median-centered Levene behavior. This is the Brown–Forsythe robust variant.

### 3.5 Fligner–Killeen

A rank-based robust homogeneity-of-variance diagnostic is also available for grouped numeric responses.

### 3.6 Multiple testing

Normality and dispersion diagnostics are corrected within explicit method/response families according to the selected `PAdjustMethod`.

---

## 4. Numeric–numeric bivariate association

`analyze_bivariate()` computes unordered pairs of eligible numeric variables.

### 4.1 Pearson correlation

Pearson product-moment correlation is calculated on pairwise finite observations. Ruddy records the number of complete and incomplete pairs.

### 4.2 Spearman correlation

Spearman rank correlation is calculated on the same pairwise finite basis.

### 4.3 Kendall correlation

Kendall rank correlation complements Pearson/Spearman for ordinal concordance and monotonic dependence.

### 4.4 Degenerate correlation cases

Correlations are not fabricated for constant variables or insufficient complete pairs. Such cases are returned with explicit status/reason values rather than silently producing an uninterpretable coefficient.

### 4.5 Multiple testing

Each correlation method defines an explicit hypothesis family. FDR-BH is the default correction for bivariate analysis.

---

## 5. Numeric–categorical comparisons

Ruddy selects the applicable test family from the method requested and the number of observed groups; it does **not** choose a method based on normality or variance diagnostics.

### 5.1 Welch independent-samples t-test

For two groups, Ruddy can run Welch's t-test (`equal_var=False`). The associated effect size is bias-corrected Hedges' g for group A minus group B.

Hedges' g uses the pooled within-group variance followed by the small-sample correction:

`J = 1 - 3 / (4 df - 1)`.

### 5.2 Mann–Whitney U

For two groups, Ruddy can run a two-sided Mann–Whitney U test. Cliff's delta is derived from U:

`delta = 2U/(n_a n_b) - 1`.

The result is bounded to `[-1, 1]`.

### 5.3 Welch one-way ANOVA

For more than two groups, Ruddy implements Welch's one-way ANOVA independently of version-specific SciPy APIs. The method uses group means, sample variances and inverse-variance weights, returning Welch F with numerator and denominator degrees of freedom.

The accompanying descriptive effect size is eta squared based on between-group/total sums of squares.

### 5.4 Kruskal–Wallis

The Kruskal–Wallis H test is available for multiple groups. The accompanying effect size is epsilon squared:

`epsilon² = (H - k + 1)/(N - k)`, bounded to `[0, 1]`.

### 5.5 Group-size and variance guards

Ruddy explicitly reports:

- insufficient group observations;
- identical pooled values;
- zero/invalid within-group variance;
- undefined effect sizes;
- non-finite inferential results.

---

## 6. Categorical–categorical associations

### 6.1 Pearson chi-square

For general contingency tables, Ruddy runs Pearson's chi-square test without silently substituting another test. It records:

- table dimensions;
- used/missing observations;
- chi-square statistic;
- degrees of freedom;
- p/q values;
- minimum expected count;
- number/fraction of expected cells below 5;
- bias-corrected Cramér's V;
- serialized contingency table.

Low expected counts are diagnostic information; they do not automatically replace chi-square.

### 6.2 Fisher exact

Fisher exact is available for 2×2 contingency tables only. Outside 2×2, the request is reported as skipped rather than generalized silently.

The 2×2 effect estimate is the odds ratio.

### 6.3 Bias-corrected Cramér's V

Ruddy uses the bias-corrected form based on corrected phi-squared and corrected table dimensions. The result is bounded to `[0, 1]`.

---

## 7. Extended dependence analysis

`analyze_dependence()` expands beyond conventional correlation.

### 7.1 Partial Pearson correlation

For numeric `X` and `Y` controlling numeric covariates `Z`, Ruddy residualizes both target variables against an intercept plus the covariate matrix and correlates the resulting residuals using Pearson correlation.

Rank-deficient covariate designs are rejected rather than regularized implicitly.

### 7.2 Partial Spearman correlation

Partial Spearman follows the same residualization framework after rank-transforming the relevant numeric columns.

### 7.3 Distance correlation

Ruddy implements biased sample distance correlation for one-dimensional variables. Pairwise Euclidean distance matrices are double-centered and combined into distance covariance/variance quantities.

The implementation is explicitly tested for:

- values in `[0, 1]`;
- `dCor(X, X) = 1` for non-degenerate X;
- detection of nonlinear relationships for which Pearson can be close to zero.

Permutation inference is available. When `n_permutations=0`, the statistic remains a valid descriptive result and p/q values remain missing rather than causing multiple-testing failure.

### 7.4 Mutual information

Ruddy estimates mutual information according to the observed data kinds:

- numeric–numeric: k-nearest-neighbor regression MI estimator;
- numeric–categorical: classification-style MI estimator;
- categorical–categorical: empirical mutual information.

Permutation inference can be used to attach p-values while preserving the MI statistic itself as the primary dependence quantity.

Mutual information is a dependence measure; Ruddy does not interpret it causally.

---

## 8. Contingency-table cell diagnostics

`analyze_contingency_diagnostics()` decomposes categorical associations beyond the global chi-square statistic.

For each cell, Ruddy reports:

- observed count;
- expected count;
- Pearson residual;
- standardized residual;
- chi-square contribution;
- fraction of total chi-square contributed by the cell;
- row fraction;
- column fraction.

This output is intended to support downstream residual heatmaps and identification of over/underrepresented level combinations.

---

## 9. Confidence intervals

`analyze_confidence_intervals()` integrates several analytic/bootstrap interval estimators.

### 9.1 Mean interval

Mean intervals use the Student-t critical value and sample standard error.

### 9.2 Pearson correlation interval

Pearson intervals use Fisher's z transformation and require `n > 3` and `|r| < 1`.

### 9.3 Spearman and Kendall intervals

For rank correlations, Ruddy uses the generic bootstrap framework rather than pretending the Pearson Fisher-z interval applies.

### 9.4 Welch mean-difference interval

The interval targets `mean(A) - mean(B)` and uses unequal-variance standard error and Welch–Satterthwaite degrees of freedom.

### 9.5 Hedges' g interval

Hedges' g confidence intervals are bootstrap-based.

### 9.6 Odds-ratio interval

For valid positive 2×2 counts, Ruddy uses a log-Wald interval. It intentionally performs **no hidden zero-cell correction**; zero cells make this interval unavailable/degenerate.

---

## 10. General bootstrap framework

`bootstrap_confidence_interval()` wraps SciPy bootstrap behavior for scalar statistics and returns a compact `BootstrapResult` rather than storing every resampled statistic.

Supported interval methods:

- percentile;
- basic;
- BCa.

The bootstrap can be paired or unpaired. The random state is explicit and deterministic.

Minimum requirements and failure states include:

- fewer than two observations in any required sample → skipped;
- non-finite original statistic → degenerate;
- bootstrap backend failure → degenerate;
- non-finite interval/SE → degenerate.

The current API requires at least 100 resamples.

---

## 11. Response-centric grouped EDA

`analyze_groups()` treats one or more selected columns as responses and evaluates them across one or more grouping variables.

### 11.1 Response catalog

Ruddy records each response's role, kind, present/missing values and finite/non-finite numeric availability.

### 11.2 Group coverage

Each grouping variable receives explicit coverage metrics, including missing group labels and observed levels.

### 11.3 Numeric responses

Within each group level Ruddy reports:

- finite response sample size;
- missing/non-finite values;
- mean and SD;
- median;
- Q25/Q75 and IQR;
- minimum/maximum.

### 11.4 Categorical responses

Within each group level Ruddy reports response-level counts and fractions.

### 11.5 Response-specific inference

Numeric responses use the configured two-group/omnibus comparison methods. Categorical responses use categorical-association methods. Each response keeps its own inferential family; multiple responses are not converted into an implicit multi-output model.

### 11.6 External annotations

Aligned annotation sources can be attached to or summarized alongside grouped analyses. Partial annotation coverage remains observable and does not trigger a global drop of all observations.

---

## 12. Post-hoc comparisons

`analyze_posthoc()` supports explicitly requested pairwise procedures for a numeric response and one factor.

### 12.1 Tukey–Kramer HSD

Ruddy computes a pooled residual mean square across groups and uses the Studentized range distribution. The Tukey–Kramer form supports unequal sample sizes through pair-specific standard errors.

Output includes group means, difference, SE, Studentized-range statistic, family-adjusted p-value and confidence interval.

### 12.2 Games–Howell

Games–Howell uses pair-specific sample variances and Welch–Satterthwaite degrees of freedom. It also uses the Studentized range distribution for p-values and intervals.

### 12.3 No automatic post-hoc choice

Ruddy never selects Tukey versus Games–Howell from a preceding homoscedasticity diagnostic. Both are explicit requests.

---

## 13. Univariate outlier and numeric-quality diagnostics

`analyze_outliers()` is non-destructive.

### 13.1 Tukey IQR fences

For finite values:

- lower fence = `Q1 - k * IQR`;
- upper fence = `Q3 + k * IQR`;
- default `k = 1.5`.

A zero IQR does not automatically invalidate the rule if the variable is non-constant; observations outside the collapsed fence can still be flagged, while the zero-IQR state is recorded separately as a quality diagnostic.

### 13.2 Modified robust Z-score

Ruddy uses:

`z* = 0.6744897501960817 * (x - median) / MAD`

with default threshold `|z*| > 3.5`.

A zero MAD makes modified Z undefined and is reported explicitly.

### 13.3 Row-level flags

Observation flags are opt-in. Summary counts are computed regardless, but row-level flag tables are returned only when `include_flags=True`.

Missing and non-finite observations are not classified as outliers.

---

## 14. Feature preprocessing

`prepare_features()` is the common feature-space preprocessing layer.

### 14.1 Finite-row requirement

Rows containing any non-finite feature value are excluded from the prepared matrix and recorded in an exclusions table with observation ID and source row index.

### 14.2 Scaling methods

Supported explicit scaling:

- `none` (default);
- `standard`;
- `robust`;
- `minmax`.

Ruddy never applies scaling automatically.

Sparse matrices are preserved where supported. Operations that would require hidden densification are rejected.

---

## 15. Principal component analysis

`analyze_pca()` provides a full PCA result rather than just plotting PC1/PC2.

Outputs include:

- observation scores;
- feature loadings;
- explained variance;
- explained variance ratio;
- cumulative explained variance ratio;
- singular values;
- exclusions;
- preprocessing metadata;
- provenance.

Requested component count is validated against effective matrix rank. A zero-rank matrix is not fitted.

`PCAResult.to_feature_matrix()` converts selected principal components into a new `FeatureMatrix` with provenance. These derived components can then be used by multivariate, group, MANOVA or representation analyses.

PCA results are marked as suitable for downstream inferential use when the user explicitly chooses to do so.

---

## 16. t-SNE and UMAP

### 16.1 t-SNE

`analyze_tsne()` supports explicit scaling, metric, perplexity, component count, maximum iterations and random seed.

### 16.2 UMAP

`analyze_umap()` supports explicit scaling, metric, neighbors, minimum distance, component count and random seed. `umap-learn` is an optional dependency; missing UMAP support raises `OptionalDependencyError` rather than breaking Ruddy imports.

### 16.3 Inferential boundary

Both nonlinear projection results are explicitly tagged:

- `exploratory = True`;
- `inferential_allowed = False`.

This is a deliberate protection against treating UMAP/t-SNE axes as stable inferential variables by default.

---

## 17. Covariance and correlation structure

`analyze_covariance_structure()` operates on a `FeatureMatrix`.

### 17.1 Common complete-case basis

The covariance and correlation matrices use one common set of rows with finite values across all selected features. This keeps the matrices internally coherent.

Separately, Ruddy reports pairwise finite counts from the source feature matrix.

### 17.2 Matrices

Outputs include:

- covariance matrix;
- Pearson correlation matrix;
- optional Spearman correlation matrix.

### 17.3 Condition diagnostics

Ruddy computes feature-level and matrix-level conditioning diagnostics on centered, sample-standardized features:

- standard deviations and constant-feature detection;
- singular values;
- effective matrix rank;
- condition number;
- condition indices.

Scale-induced condition problems are reduced by standardizing specifically for this diagnostic basis.

---

## 18. Multicollinearity: VIF and tolerance

`analyze_collinearity()` computes variance inflation factor (VIF) and tolerance for dense feature matrices.

Each auxiliary regression includes an intercept. The intercept is not reported as a feature.

Ruddy computes VIF only when:

- at least two non-constant features exist;
- the standardized design is full rank;
- residual degrees of freedom are available.

Perfect collinearity is returned as a rank-deficient design, rather than reporting apparently meaningful infinite VIF values while continuing as normal.

Condition diagnostics are returned alongside feature-level VIF/tolerance.

---

## 19. Mahalanobis diagnostics

`analyze_mahalanobis()` supports classical and optional robust multivariate distances.

### 19.1 Classical Mahalanobis

The classical estimate uses sample center/covariance and requires an invertible covariance structure. Ruddy does not silently use a pseudo-inverse when covariance is singular.

### 19.2 Robust Mahalanobis

The robust method uses scikit-learn `MinCovDet` and can accept an explicit support fraction.

### 19.3 Reference threshold

Flags use a chi-square reference threshold with degrees of freedom equal to the number of analyzed features. The default quantile is 0.975.

The flag is diagnostic only; observations are never removed.

### 19.4 High-dimensional guard

If dimensionality relative to observations makes covariance inversion indefensible, Ruddy reports the analysis as not feasible. The expected workflow is explicit dimensionality reduction, typically PCA, before Mahalanobis analysis.

---

## 20. Combined multivariate EDA

`analyze_multivariate()` is a convenience composition of:

1. covariance/correlation structure;
2. collinearity diagnostics;
3. classical/robust Mahalanobis diagnostics.

It does not add new mathematical logic beyond those standalone methods.

---

## 21. MANOVA

`analyze_manova()` fits a main-effects multivariate analysis of variance/covariance using explicitly selected:

- multiple numeric responses;
- categorical factors;
- optional numeric covariates.

Reported multivariate statistics include:

- Wilks' lambda;
- Pillai's trace;
- Hotelling–Lawley trace;
- Roy's greatest root.

For each term/statistic, Ruddy reports the statistic, F approximation, numerator/denominator degrees of freedom and p-value.

### 21.1 Guards

MANOVA checks:

- at least two responses;
- maximum response count;
- factor-level cardinality and minimum level size;
- complete-case observations;
- response-matrix rank;
- design-matrix rank;
- residual degrees of freedom.

Rank-deficient response spaces or model designs are not silently regularized.

Interactions are intentionally handled by the factorial engine rather than the current MANOVA layer.

---

## 22. PERMANOVA and PERMDISP

`analyze_permutation_group_structure()` always runs PERMANOVA and PERMDISP together on the same aligned feature-space grouping factor.

### 22.1 PERMANOVA

Ruddy constructs a pairwise distance matrix using the requested SciPy distance metric and computes a pseudo-F statistic from between/within sums of squares. It reports:

- pseudo-F;
- R² (fraction of distance-space variation associated with group labels);
- between/within degrees of freedom;
- permutation p-value.

### 22.2 PERMDISP

The same distance matrix is mapped to principal-coordinate space (PCoA), group centroids are estimated there, and observation distances to group centroids are used for an F-style dispersion comparison with permutation inference.

### 22.3 Negative PCoA eigenvalues

For non-Euclidean distance structures, negative PCoA eigenvalues may occur. Ruddy records this as an advisory rather than silently correcting the geometry.

### 22.4 Interpretation

PERMANOVA separation should be considered alongside PERMDISP, because apparent group separation may coexist with or be influenced by heterogeneous dispersion.

---

## 23. Factorial ANOVA/ANCOVA

`analyze_factorial()` fits an explicit OLS factorial model.

### 23.1 Supported model terms

Ruddy supports:

- main factor effects;
- numeric covariates;
- factor/factor interactions;
- factor/covariate interactions when explicitly requested;
- interactions up to a configurable maximum order.

Formula shorthand supports patterns such as:

```text
Y ~ A
Y ~ A + B
Y ~ A:B
Y ~ A * B
Y ~ A * B + X
Y ~ A * B * C
```

`A * B` expands hierarchically to `A + B + A:B`.

The formula parser deliberately does not evaluate arbitrary functions/code such as `log(x)` or `I(x**2)`. Transformations should exist as explicit dataset columns.

### 23.2 Type II and Type III sums of squares

The user explicitly selects Type II or Type III sums of squares. Ruddy never chooses based on diagnostics or balance.

Factors use sum-to-zero contrasts. Type III therefore does not depend on an arbitrary treatment-reference level; the same contrast policy is retained under Type II so changing SS type does not silently change coding.

### 23.3 Robust covariance

The user may explicitly request HC0, HC1, HC2 or HC3 covariance. Heteroscedasticity diagnostics do not enable robust covariance automatically.

### 23.4 Factorial effect sizes

For each estimable term Ruddy reports:

- eta squared;
- partial eta squared;
- omega squared;
- partial omega squared.

### 23.5 Cell accounting

The factorial cell table includes the full combination structure, including empty combinations. Ruddy records:

- empty cells;
- cells below `min_cell_n`;
- balanced/unbalanced design status.

Empty/small cells become diagnostics/advisories and may also make the design rank-deficient.

### 23.6 Rank and degrees-of-freedom guards

Ruddy rejects or skips models with:

- constant response;
- no factor terms where a factorial effect is required;
- rank-deficient design;
- insufficient residual degrees of freedom;
- non-estimable terms.

No hidden pseudo-inverse is used to disguise non-estimability.

---

## 24. Factorial model diagnostics

`FactorialResult` contains both aggregate diagnostics and observation-level influence diagnostics.

### 24.1 Residual normality

The model diagnostics include:

- Shapiro–Wilk when sample size is within the configured limit;
- Jarque–Bera.

### 24.2 Heteroscedasticity

Breusch–Pagan is computed when estimable.

### 24.3 Residual dispersion by factorial cells

Median-centered Levene/Brown–Forsythe style diagnostics can compare residual dispersion across observed factorial cells.

### 24.4 Design conditioning

The model design condition number is reported and can be flagged relative to the configured threshold.

### 24.5 Observation-level influence

For complete-case observations Ruddy reports:

- fitted value;
- residual;
- studentized residual;
- leverage;
- Cook's distance;
- large-residual flag;
- high-leverage flag;
- influential flag.

Default diagnostic thresholds are recorded in the output so downstream applications do not need to guess how a flag was defined.

### 24.6 Diagnostics do not mutate the model

A diagnostic can be flagged while the requested model remains exactly the requested model. This is a central Ruddy policy.

---

## 25. Estimated marginal means and contrasts

`analyze_marginal_means()` builds model-based marginal estimates from an OLS factorial/ANCOVA design.

### 25.1 Weighting policy

For a requested factor term:

- covariates are held at their observed mean;
- nuisance factors are averaged with **equal weighting across factor levels**.

This weighting policy is stored in provenance.

### 25.2 Means

Each estimated marginal mean reports estimate, standard error, residual df and confidence interval.

### 25.3 Pairwise contrasts

Pairwise contrasts are linear combinations of the same fitted model, not independent raw-data tests. Ruddy reports estimate difference, SE, t, p/q and CI.

FDR-BH can be applied explicitly to contrast families.

---

## 26. Mixed-effects models

`analyze_mixed_effects()` implements a deliberately scoped linear mixed model:

- one grouping factor;
- random intercept;
- optional random slopes for explicitly declared numeric covariates;
- fixed factor/covariate/interactions through the fixed-effects design.

Ruddy does not expose an unrestricted mixed-model DSL.

### 26.1 Output

`MixedEffectsResult` contains:

- fixed-effect estimates, SE, z, p and CI;
- random-effect covariance/variance components;
- conditional random effects by group;
- excluded observations;
- model summary;
- advisories and provenance.

### 26.2 Model summary

The summary tracks convergence, REML/ML choice, likelihood criteria, residual variance, random-intercept variance and an intercept ICC when available.

### 26.3 Convergence and singularity

Non-convergence and singular random-effect covariance are observable states/advisories. Failed fits are not presented as successful models.

---

## 27. Representation-space alignment

`align_feature_matrices()` aligns two dense numerical spaces by observation ID before any representation comparison.

Strict alignment requires exact ID coverage; partial alignment retains only common observations while exposing missing/unmatched IDs.

Rows that are aligned by ID but contain non-finite features in either space are excluded jointly and recorded.

Representation comparison currently requires dense inputs; Ruddy does not silently densify sparse matrices.

---

## 28. Canonical correlation analysis (CCA)

CCA is implemented through scikit-learn after explicit scaling (standard by default in representation comparison).

Ruddy computes effective ranks of both spaces and limits the number of canonical components to:

`min(rank(X), rank(Y), n-1, p_X, p_Y)`.

If the requested component count exceeds this effective limit, CCA is returned as skipped.

Outputs include:

- canonical correlations;
- squared canonical correlation (`shared_variance` field);
- X/Y canonical weights;
- X/Y loadings;
- X/Y canonical scores aligned by observation ID.

CCA fit failures are represented explicitly with an advisory.

---

## 29. Linear centered kernel alignment (CKA)

`linear_cka()` centers both feature matrices by feature mean and computes the normalized squared Frobenius norm of the cross-covariance structure:

`CKA = ||XcᵀYc||²_F / (||XcᵀXc||_F ||YcᵀYc||_F)`.

Important properties tested in Ruddy include:

- `0 <= CKA <= 1`;
- identical non-degenerate representations yield 1;
- orthogonal rotation of a representation preserves linear CKA;
- X and Y may have different numbers of features.

CKA is undefined for zero-variance representation spaces.

---

## 30. Procrustes representation comparison

For spaces with the same feature dimensionality, Ruddy:

1. centers each representation;
2. normalizes each by its Frobenius norm;
3. computes an orthogonal Procrustes mapping;
4. reports squared disparity after alignment;
5. reports `similarity = max(0, 1 - disparity)`.

When dimensions differ, Procrustes is returned as skipped rather than implicitly reducing either space.

---

## 31. Distance-space similarity

Ruddy computes all condensed pairwise observation distances separately in the two aligned representation spaces using the requested SciPy distance metric.

The two distance vectors are compared using either Pearson or Spearman correlation. Output includes coefficient, p-value and number of observation pairs.

This asks whether the two representations preserve similar **pairwise geometry**, not whether their feature coordinates are directly aligned.

---

## 32. Mantel permutation test

Ruddy constructs square pairwise-distance matrices for both representation spaces. The observed Mantel statistic is Pearson correlation between the upper-triangular distance vectors.

Permutation inference permutes the observation labels of one distance matrix and uses a two-sided absolute-statistic exceedance rule with the standard `(exceed + 1)/(permutations + 1)` finite-permutation correction.

When zero permutations are requested, the correlation is still returned while the p-value remains missing.

---

## 33. Compositional data analysis

Compositional methods require dense finite non-negative data with positive row sums.

### 33.1 Closure

`closure()` rescales each row so its components sum to the requested total (1.0 by default).

### 33.2 Multiplicative zero replacement

`multiplicative_zero_replacement()` is explicit and opt-in. Ruddy never replaces zeros simply because a log-ratio transform was requested.

The replacement fraction must lie strictly between 0 and 1. Replacement details are returned so affected observations remain traceable.

### 33.3 CLR

Centered log-ratio (CLR) transforms each closed positive composition using log component values centered by the row's mean log value.

### 33.4 ALR

Additive log-ratio (ALR) expresses all non-denominator components relative to a selected denominator component. The denominator index is explicit (`-1` by default).

### 33.5 ILR

Isometric log-ratio (ILR) uses an orthonormal Helmert basis to map a D-part composition to `D-1` Euclidean coordinates.

### 33.6 Variation matrix

The variation matrix contains the variance of pairwise log-ratios between composition components.

### 33.7 Aitchison distances

Ruddy computes Euclidean distances in CLR space and returns a labeled observation × observation matrix.

### 33.8 Integrated compositional result

`analyze_composition()` returns:

- transformed `FeatureMatrix`;
- variation matrix;
- Aitchison distance matrix;
- zero-replacement report;
- provenance.

The transformed feature matrix can feed downstream PCA, multivariate or representation analyses.

---

## 34. Bayesian exploratory estimation

`analyze_bayesian_eda()` is a scoped conjugate analysis for numeric means, not a general Bayesian modeling system.

### 34.1 Prior/model

For each numeric variable Ruddy uses a Normal–Inverse-Gamma prior for unknown mean/variance with configurable hyperparameters. Defaults are deliberately weak:

- prior mean = 0;
- kappa = `1e-6`;
- alpha = `1e-6`;
- beta = `1e-6`.

Posterior mean draws are generated by sampling posterior variance from the inverse-gamma distribution and then the mean conditional on that variance.

### 34.2 Posterior summaries

Ruddy reports:

- posterior mean;
- posterior median;
- equal-tailed credible interval;
- probability positive;
- probability negative;
- probability of direction (`max(P>0, P<0)`);
- ROPE probability.

### 34.3 Binary-group differences

For a requested grouping column with exactly two levels, separate posterior mean draws are obtained for each group and differenced. The same posterior summary is reported for `mean(A) - mean(B)`.

Grouping variables with other than two levels are skipped for Bayesian mean-difference output.

### 34.4 Reproducibility

Posterior sampling uses an explicit NumPy random generator seeded by `random_state`.

---

## 35. Multivariate anomaly diagnostics

`analyze_anomalies()` provides two explicitly ML-based diagnostics without changing the underlying data.

### 35.1 Isolation Forest

Ruddy fits scikit-learn Isolation Forest with explicit estimator count, contamination and random state. It transforms `score_samples` so larger `anomaly_score` means more anomalous.

### 35.2 Local Outlier Factor

Ruddy fits scikit-learn Local Outlier Factor with explicit neighbor count and contamination. The negative outlier factor is sign-flipped so larger Ruddy scores again mean more anomalous.

LOF is skipped when `n_neighbors` is not less than the number of analyzed observations.

### 35.3 Preprocessing and exclusions

Anomaly methods use the common feature-preparation layer, so non-finite rows are excluded explicitly and optional scaling is recorded.

### 35.4 Interpretation

An anomaly flag means only that a specified method regards an observation as unusual under its own geometry. Ruddy does not label the observation erroneous or remove it.

---

## 36. Unified analysis orchestration

`analyze()` accepts a `TabularDataset`, optional primary `FeatureMatrix`, optional comparison `FeatureMatrix`, optional aligned annotations and an `AnalysisConfig`.

The orchestrator contains **no new statistical logic**. Each enabled block delegates to the same standalone function documented above.

Default enabled blocks are:

- profiling;
- univariate.

Everything else is opt-in.

This design supports two important guarantees:

1. standalone API and unified API should produce scientifically equivalent component results;
2. downstream applications can reason about which result components exist by inspecting `executed_blocks` and `UnifiedAnalysisResult` fields.

---

## 37. Methods deliberately not implemented in the current MVP

The following methods are outside the scope of the first release:

- PLS;
- clustering algorithms;
- repeated-measures-specific procedures;
- survival analysis;
- time-series analysis;
- quantile regression;
- unrestricted generalized/mixed model specifications;
- arbitrary Bayesian graphical/hierarchical model builders.

Their absence is deliberate and should not be confused with undocumented functionality.
