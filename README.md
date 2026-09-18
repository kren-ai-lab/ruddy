# Ruddy

**Ruddy** is a domain-agnostic Python library for statistical exploratory data analysis (EDA) of tabular datasets and numerical feature spaces. It is designed for scientific workflows in which observations may be described by conventional table columns, dense or sparse feature matrices, embeddings, descriptor vectors, structural encodings, or other numerical representations.

Ruddy separates **scientific computation** from **visual rendering**. The core library produces structured, traceable results; visualization is demonstrated externally in the example notebooks and is intentionally not a dependency of the scientific core.

> Current status: scientific MVP feature-complete and tested. Product hardening, configuration files, and a local visual application remain future productization work.

## What Ruddy provides

Ruddy currently implements the following analysis families.

| Area | Implemented capabilities |
| --- | --- |
| Data contracts | `TabularDataset`, `FeatureMatrix`, statistical roles/kinds, strict/partial ID alignment, external annotations |
| Profiling | dataset overview, column profiling, missingness, pairwise completeness, missingness patterns |
| Univariate EDA | numeric, categorical, boolean and datetime summaries; entropy; quantiles; robust summaries |
| Distribution diagnostics | Shapiro–Wilk, D'Agostino K², Anderson–Darling, Brown–Forsythe, Fligner–Killeen |
| Bivariate EDA | Pearson, Spearman, Kendall; Welch t, Mann–Whitney, Welch ANOVA, Kruskal–Wallis; χ², Fisher exact |
| Effect sizes | Hedges' g, Cliff's delta, η², ε², bias-corrected Cramér's V, factorial η²/partial η²/ω²/partial ω² |
| General dependence | partial Pearson/Spearman, distance correlation, mutual information, permutation inference |
| Confidence intervals | means, Pearson correlations, mean differences, Hedges' g, odds ratios, bootstrap intervals |
| Groups | response-centric summaries and inferential comparisons across factors/groups, aligned annotations |
| Post-hoc | Tukey–Kramer HSD and Games–Howell |
| Outliers | Tukey IQR, modified robust Z-score, explicit data-quality diagnostics |
| Projections | PCA, t-SNE, optional UMAP, explicit scaling, row-exclusion accounting |
| Multivariate EDA | covariance/correlation structure, VIF/tolerance, condition diagnostics, classical/robust Mahalanobis |
| MANOVA | main-effects MANOVA with Wilks, Pillai, Hotelling–Lawley and Roy statistics |
| Permutation group analysis | PERMANOVA + PERMDISP using a shared distance space |
| Factorial modeling | ANOVA/ANCOVA, interactions, Type II/III SS, robust HC covariance options, model diagnostics |
| Marginal means | equal-weight estimated marginal means and model-based pairwise contrasts |
| Mixed effects | random-intercept models with optional numeric random slopes |
| Representation comparison | CCA, linear CKA, Procrustes, distance-space similarity, Mantel permutation test |
| Compositional data | closure, multiplicative zero replacement, CLR, ALR, ILR, variation matrix, Aitchison distance |
| Bayesian EDA | posterior means, binary-group mean differences, credible intervals, probability of direction, ROPE |
| Multivariate anomaly diagnostics | Isolation Forest and Local Outlier Factor |
| Orchestration | explicit block-based `ruddy.analyze()` pipeline with common configuration and provenance |
| CLI | grouped commands for all major scientific blocks plus the unified `ruddy pipeline` |
| Visualization examples | 13 marimo examples demonstrating rich static and interactive visualizations outside the core |

## Design principles

Ruddy is intentionally conservative about scientific automation.

- **No silent coercion.** Numeric-looking strings are not silently converted to numeric data.
- **No silent row deletion.** Complete-case or finite-row exclusions are recorded explicitly whenever an analysis requires them.
- **No hidden scaling.** Feature scaling is always requested explicitly; the default is `none`.
- **No hidden dimensionality reduction.** High-dimensional methods fail or skip explicitly when their mathematical requirements are not met.
- **No automatic test switching.** Distribution or variance diagnostics never change the inferential method requested by the user.
- **No hidden pseudo-inverse/regularization.** Rank-deficient covariance, CCA, factorial, MANOVA and related problems are surfaced as explicit degenerate/skipped states.
- **No destructive outlier handling.** Outliers/anomalies are flagged, never removed or modified.
- **Explicit multiple-testing families.** FDR correction is performed only inside declared hypothesis families and only for finite inferential p-values.
- **Identity before row position.** Tabular data, annotations and representation matrices are aligned by observation identifiers.
- **Structured provenance.** Analysis parameters, input summaries, random seeds and Ruddy version are retained in result contracts.

## Installation

Ruddy currently targets Python 3.11–3.14.

```bash
python -m pip install -e .
```

Core dependencies are NumPy, pandas, Polars, pyarrow, SciPy, scikit-learn, statsmodels, Typer and Rich. pandas supports accepted inputs and the explicit statsmodels/Patsy model boundaries; public result tables use Polars.

UMAP is optional:

```bash
python -m pip install -e ".[manifold]"
```

The visualization examples deliberately use dependencies outside the core:

```bash
uv sync --group examples
```

## Minimal tabular workflow

```python
import polars as pl
from ruddy import TabularDataset, analyze_univariate, analyze_bivariate

frame = pl.read_csv("data.csv")

dataset = TabularDataset(
    frame,
    id_column="id",
    role_overrides={
        "activity": "response",
        "family": "factor",
        "length": "covariate",
        "sequence": "excluded",
    },
)

univariate = analyze_univariate(dataset)
bivariate = analyze_bivariate(dataset)

print(univariate.numeric_statistics.head())
print(bivariate.correlations.head())
```

Result tables are returned as Polars DataFrames (`pl.DataFrame`). pandas `DataFrame` objects are also accepted directly as inputs to `TabularDataset` and `FeatureMatrix`.

The statistical role of a column is separate from its observed data kind. A numerically encoded batch column can therefore be explicitly declared as a factor and will be treated categorically by grouped/factorial methods.

## Minimal feature-space workflow

```python
import numpy as np
import polars as pl
from ruddy import FeatureMatrix, analyze_pca, analyze_multivariate

matrix = np.load("embedding.npy")
ids = pl.read_csv("ids.csv")["id"]

features = FeatureMatrix(matrix, observation_ids=ids)

pca = analyze_pca(
    features,
    n_components=20,
    scaling="standard",
)

multivariate = analyze_multivariate(
    pca.to_feature_matrix(),
    scaling="none",
)
```

PCA components can be converted back into a `FeatureMatrix` with provenance and used by downstream multivariate analyses. UMAP and t-SNE outputs are explicitly marked as exploratory and are not treated as inferential variables by design.

## Comparing two numerical representations

```python
from ruddy import FeatureMatrix, analyze_representation_similarity

space_a = FeatureMatrix(embedding_a, observation_ids=ids_a)
space_b = FeatureMatrix(descriptors_b, observation_ids=ids_b)

comparison = analyze_representation_similarity(
    space_a,
    space_b,
    alignment="strict",
    cca_components=5,
    cca_scaling="standard",
    distance_metric="cosine",
    mantel_permutations=999,
    random_state=42,
)

print(comparison.cka)
print(comparison.cca.correlations)
print(comparison.mantel)
```

The two spaces do not need the same number of features for CKA, CCA or distance-space comparison. Procrustes is reported as skipped when dimensionality is not directly compatible.

## Unified analysis

`ruddy.analyze()` is an orchestrator, not a separate statistical implementation. It delegates to the same standalone public methods.

```python
from ruddy import AnalysisConfig, analyze

config = AnalysisConfig(
    enabled_blocks=(
        "profiling",
        "univariate",
        "bivariate",
        "groups",
        "outliers",
    ),
    responses=("activity",),
    groups=("family",),
    random_state=42,
)

result = analyze(dataset, config=config)
```

The default `AnalysisConfig()` enables only `profiling` and `univariate`. Inferential, multivariate and representation analyses are never activated implicitly.

## Command-line interface

```bash
ruddy --help
ruddy --version
```

Commands are grouped by intent:

```text
ruddy pipeline   the unified run; blocks selected with --enable
ruddy inspect    profile · univariate · diagnostics · outliers · multivariate
ruddy analyze    bivariate · dependence · contingency · intervals ·
                 groups · posthoc · representation · compositional ·
                 bayesian · anomaly
ruddy model      factorial · marginal-means · mixed-effects · manova · permanova
ruddy project    pca · tsne · umap
```

For example:

```bash
ruddy analyze bivariate data.csv \
  --id-column id \
  --factor family \
  --exclude sequence \
  --output-dir results
```

or:

```bash
ruddy project pca embeddings.csv \
  --id-column id \
  --n-components 20 \
  --scaling standard \
  --output-dir results
```

Each command closes with a run summary on stderr and reports user errors
without a traceback, exiting with code `2`. Commands that emit JSON keep
stdout clean for piping. See `docs/CLI_REFERENCE.md` for the full reference.

Configuration files and progress reporting remain future productization work rather than part of the scientific core.

## Results and scientific states

Many analyses use the explicit scientific status contract:

- `ok` — the requested statistic/model was estimable.
- `degenerate` — the requested analysis is defined conceptually but the observed data make the statistic/model degenerate, for example constant variables or singular covariance.
- `skipped` — requirements for the requested analysis were not satisfied, for example insufficient observations or excessive effective dimensionality.

A `reason` accompanies non-`ok` states, and non-fatal scientific conditions are reported through advisories. Missing values, non-finite values, excluded rows, rank problems and coverage mismatches are designed to remain observable rather than being converted into generic `NaN` outputs.

## Visualization examples

The scientific core contains no Matplotlib or Plotly dependency. Rich visualization demonstrations live under `examples/`.

Each example is a [marimo](https://marimo.io) notebook stored as a plain Python file. Run it as a script, or open it interactively:

```bash
uv sync --group examples
MPLBACKEND=Agg uv run python examples/01_profiling_univariate.py   # script
uv run marimo edit examples/01_profiling_univariate.py             # notebook
```

The current gallery covers distributions by group, nonlinear dependence, post-hoc comparisons, effect sizes, factorial interactions, marginal means, PCA/t-SNE, PERMANOVA/PERMDISP, CKA/CCA/Procrustes/Mantel, compositional geometry, anomaly-method agreement, Bayesian uncertainty and an end-to-end numerical-representation workflow.

See [examples/README.md](examples/README.md) and [docs/VISUALIZATION_EXAMPLES.md](docs/VISUALIZATION_EXAMPLES.md).

## Breaking changes (Polars migration)

Ruddy migrated its internal tabular engine and public result tables to Polars:
- **Polars result tables**: Every analysis returns Polars DataFrames (`pl.DataFrame`) with declared schemas.
- **`TabularDataset` surface**: Stored table is exposed via `.frame` (`pl.DataFrame`); legacy adapter methods (frame export, `select()`, `n_observations`, `n_columns`, `columns`, `__len__`, and `observation_id_tuple`) were removed in favor of `dataset.frame.height`, `dataset.frame.width`, `dataset.frame.columns`, and `dataset.frame.select(...)`.
- **pandas index is never identity**: A pandas DataFrame with a non-default index requires explicit `observation_ids` or `reset_index()`. In `align_annotations`, `id_column` is strictly required.
- **Complex dtypes rejected**: Unsupported complex dtypes raise `TypeError` at dataset construction.
- **Null instead of NaN**: Missing values in result tables are represented as Polars `null`, never float `NaN`.
- **Labeled matrix layout**: Square and labeled matrices contain row labels in the first column (`feature` or `observation_id`) retaining label/ID dtypes, exported without an index column.
- **`ColumnSpec.dtype` strings**: Schema metadata records Polars type names (e.g., `"Int64"`, `"Float64"`, `"String"`).

See [docs/CHANGELOG_POLARS.md](docs/CHANGELOG_POLARS.md) for full details.

## Documentation

Detailed technical documentation is available in [`docs/`](docs/README.md):

- [Scientific scope](docs/SCIENTIFIC_SCOPE.md)
- [Data contracts and alignment](docs/DATA_CONTRACTS.md)
- [Complete method reference](docs/METHODS.md)
- [Method inventory](docs/METHOD_INVENTORY.md)
- [Statistical policies](docs/STATISTICAL_POLICIES.md)
- [Result contracts and provenance](docs/RESULTS_AND_PROVENANCE.md)
- [Output schema reference](docs/OUTPUT_SCHEMAS.md)
- [Unified analysis and configuration](docs/UNIFIED_ANALYSIS.md)
- [CLI reference](docs/CLI_REFERENCE.md)
- [Visualization examples](docs/VISUALIZATION_EXAMPLES.md)
- [Testing and feature freeze](docs/TESTING_AND_REPRODUCIBILITY.md)
- [Public Python API inventory](docs/PUBLIC_API.md)
- [Polars migration changelog](docs/CHANGELOG_POLARS.md)

## Validation

The scientific core is covered by unit, integration, parity, pathological-data and reproducibility tests. To validate a working checkout:

```bash
pytest -q
```

The examples are executed in CI by `examples/run_ci_examples.sh`.

## Scope boundary

Ruddy does **not** generate biological/molecular representations and does not assign domain-specific meaning to observations. It operates on tabular variables and numerical feature spaces. Likewise, the current core does not create plots or reports directly; it returns structured outputs that downstream notebooks or a future local visual interface can render.

Ruddy is currently being developed as the EDA and visual-analysis layer of a broader numerical-representation workflow, while remaining usable as a standalone domain-agnostic statistical library.
