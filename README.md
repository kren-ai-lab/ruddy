# Ruddy

**Ruddy** is a Python library for statistical exploratory data analysis of
tabular datasets and numerical feature spaces: embeddings, descriptors and
other observation-aligned matrices. It returns Polars tables with sample counts,
scientific diagnostics and provenance, ready for filtering, export and plotting.

Ruddy is domain-agnostic. It analyzes supplied data and representations; it does
not generate domain-specific features. Plotting lives in the
[example notebooks](examples/README.md), outside the scientific core.

## Installation

Ruddy is in pre-release development and supports Python 3.11–3.14.
From a checkout:

```bash
python -m pip install -e .
```

For optional UMAP support:

```bash
python -m pip install -e ".[manifold]"
```

For development setup with `uv`, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Analyze a table

```python
import polars as pl
from ruddy import TabularDataset, analyze_univariate, analyze_bivariate

frame = pl.DataFrame({
    "id": ["s1", "s2", "s3", "s4", "s5", "s6"],
    "activity": [1.2, 2.4, 1.8, 3.1, 2.9, 4.2],
    "length": [10.0, 14.0, 12.0, 17.0, 15.0, 20.0],
    "family": ["A", "A", "A", "B", "B", "B"],
})
dataset = TabularDataset(
    frame,
    id_column="id",
    role_overrides={
        "activity": "response",
        "family": "factor",
        "length": "covariate",
    },
)

univariate = analyze_univariate(dataset)
bivariate = analyze_bivariate(dataset)
print(univariate.numeric_statistics)
print(bivariate.correlations)
```

Use `pl.read_csv(...)` or `pl.read_parquet(...)` for file inputs. pandas
DataFrames are also accepted. Column roles are separate from storage types:
a numerically encoded batch can be explicitly declared as a categorical factor.
See [data contracts](docs/DATA_CONTRACTS.md) for identity and alignment rules.

## Analyze a feature space

```python
from ruddy import FeatureMatrix, analyze_pca

features = FeatureMatrix(
    dataset.frame.select("activity", "length"),
    observation_ids=dataset.observation_ids,
)
pca = analyze_pca(features, n_components=2, scaling="standard")
print(pca.scores)
print(pca.variance)
```

`FeatureMatrix` also accepts NumPy arrays and SciPy sparse matrices, with
method-specific sparse support. To compare two spaces, supply explicit IDs:

```python
# Given two FeatureMatrix objects named space_a and space_b:
from ruddy import analyze_representation_similarity

comparison = analyze_representation_similarity(
    space_a,
    space_b,
    alignment="strict",
    cca_components=2,
    random_state=42,
)
print(comparison.cka)
print(comparison.mantel)
```

See the [method reference](docs/METHODS.md) for supported analyses, assumptions
and output interpretation. It covers descriptive statistics, group comparisons,
dependence, uncertainty, factorial/mixed models, projections, multivariate
structure, representation comparison, compositions, Bayesian EDA and anomalies.

## Run several analyses

```python
from ruddy import AnalysisConfig, analyze

result = analyze(
    dataset,
    config=AnalysisConfig(
        enabled_blocks=("profiling", "univariate", "groups", "outliers"),
        responses=("activity",),
        groups=("family",),
        random_state=42,
    ),
)
print(result.summary())
```

The default configuration enables only profiling and univariate analysis.
See [unified analysis](docs/UNIFIED_ANALYSIS.md) for block selection and parameters.

## Command line

```bash
ruddy --help
ruddy analyze bivariate data.csv \
  --id-column id \
  --factor family \
  --output-dir results
```

Commands are grouped under `inspect`, `analyze`, `model` and `project`;
`ruddy pipeline` runs explicitly selected blocks together. See the
[CLI reference](docs/CLI_REFERENCE.md) for inputs, options and exported files.

## Design principles

- Data transformations, scaling and dimensionality reduction are explicit.
  Missing values are not imputed and numeric-looking strings are not coerced.
- Analyses report the observations they use; exclusions do not delete source rows.
- Diagnostics do not change the requested test. Rank-deficient problems are
  reported instead of silently regularized.
- Outliers and anomalies are flagged, never removed or modified.
- Data and annotations align by observation ID, not row position.
- Results retain parameters, sample summaries, seeds and the Ruddy version.

Check `status` and `reason` alongside estimates. `ok` means estimable;
`degenerate` means the data make the quantity non-estimable; `skipped` means a
requirement was not met. Unavailable table values are Polars nulls.
See [results and provenance](docs/RESULTS_AND_PROVENANCE.md) for reading and
exporting results.

## Documentation and examples

The [documentation index](docs/README.md) links the user guides and development
instructions. The [13 example notebooks](examples/README.md) demonstrate complete
workflows and plots. Open one interactively with:

```bash
uv sync --all-extras --group examples
uv run marimo edit examples/01_profiling_univariate.py
```

Ruddy does not provide supervised prediction, clustering, survival/time-series
models, repeated-measures-specific procedures or unrestricted GLM/Bayesian model
specification. It returns statistical evidence; domain interpretation, causal
claims and decisions about flagged observations remain with the caller.
