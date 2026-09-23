# Ruddy

[![PyPI](https://img.shields.io/pypi/v/ruddy?style=flat-square)](https://pypi.org/project/ruddy/)
[![PyVersions](https://img.shields.io/pypi/pyversions/ruddy?style=flat-square)](https://github.com/kren-ai-lab/ruddy)
[![Tests](https://img.shields.io/github/actions/workflow/status/kren-ai-lab/ruddy/tests.yml?style=flat-square)](https://github.com/kren-ai-lab/ruddy/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/kren-ai-lab/ruddy?style=flat-square)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22920317-blue?style=flat-square)](https://doi.org/10.5281/zenodo.22920317)

Ruddy is a Python library for statistical exploratory data analysis. It works
on tables and on numerical feature spaces such as embeddings or descriptor
matrices, as long as each row belongs to an identifiable observation. Results
come back as Polars tables that carry sample counts, diagnostics and the
parameters used, so you can filter them, export them or plot them.

Ruddy doesn't know about any particular domain and doesn't compute features. You
bring the data or the representation, and Ruddy analyzes it. It also has no
plotting code, but you can use your prefered library for visualizations.
See [example notebooks](https://github.com/kren-ai-lab/ruddy/blob/main/examples/README.md) for more comprehensive examples.

## Installation

Ruddy supports Python 3.11 to 3.14.

```bash
python -m pip install ruddy
```

For optional UMAP support:

```bash
python -m pip install "ruddy[manifold]"
```

For development setup with `uv`, see [DEVELOPMENT.md](https://github.com/kren-ai-lab/ruddy/blob/main/DEVELOPMENT.md).

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

To read files, use `pl.read_csv(...)` or `pl.read_parquet(...)`. pandas
DataFrames work too. A column's role doesn't depend on its dtype, so a batch
stored as integers can still be declared a categorical factor. The
[data contracts](https://github.com/kren-ai-lab/ruddy/blob/main/docs/DATA_CONTRACTS.md) guide explains how observation IDs
are checked and aligned.

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

`FeatureMatrix` also takes NumPy arrays and SciPy sparse matrices. Not every
method accepts sparse input, and Ruddy raises an error rather than densify it
behind your back. To compare two feature spaces, give both of them observation
IDs:

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

The [method reference](https://github.com/kren-ai-lab/ruddy/blob/main/docs/METHODS.md) lists every analysis with its
assumptions and how to read its output. Besides the ones above, Ruddy has group
comparisons, bootstrap intervals, factorial and mixed-effects models, PERMANOVA,
compositional data, Bayesian summaries and anomaly flags.

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

By default only profiling and univariate analysis run. The
[unified analysis](https://github.com/kren-ai-lab/ruddy/blob/main/docs/UNIFIED_ANALYSIS.md) guide covers the other blocks and
their parameters.

## Command line

```bash
ruddy --help
ruddy analyze bivariate data.csv \
  --id-column id \
  --factor family \
  --output-dir results
```

The commands are grouped under `inspect`, `analyze`, `model` and `project`.
`ruddy pipeline` runs several blocks in one go, but only the ones you select.
The [CLI reference](https://github.com/kren-ai-lab/ruddy/blob/main/docs/CLI_REFERENCE.md) documents every option and the
files each command writes.

## Design principles

Ruddy tries hard not to make decisions for you:

- Scaling and dimensionality reduction happen only when you ask for them.
  Missing values aren't imputed, and a string that looks like a number stays a
  string.
- Each analysis reports which observations it used. Excluding a row never
  deletes it from your data.
- A diagnostic never swaps the test you asked for. If a normality check fails,
  you still get the test you requested, with the diagnostic next to it.
- A rank-deficient problem is reported as such, not quietly regularized.
- Outliers are flagged and left alone.
- Data and annotations are matched by observation ID, never by row position.
- Every result records its parameters, input sizes, random seed and the Ruddy
  version that produced it.

Because of this, read `status` and `reason` next to every estimate. `ok` means
the quantity was estimated. `degenerate` means the data can't support it, for
example a constant column in a correlation. `skipped` means a requirement
wasn't met. Missing values in result tables are Polars nulls, not NaN. See
[results and provenance](https://github.com/kren-ai-lab/ruddy/blob/main/docs/RESULTS_AND_PROVENANCE.md) for how to read and
export results.

## Documentation and examples

The [documentation index](https://github.com/kren-ai-lab/ruddy/blob/main/docs/README.md) links all the guides. There are also
[13 example notebooks](https://github.com/kren-ai-lab/ruddy/blob/main/examples/README.md) written in marimo. To open one:

```bash
uv sync --all-extras --group examples
uv run marimo edit examples/01_profiling_univariate.py
```

## Citing and license

If you use Ruddy in published work, please cite it using
[`CITATION.cff`](https://github.com/kren-ai-lab/ruddy/blob/main/CITATION.cff), or the "Cite this repository" button on
GitHub. Ruddy is released under the [MIT license](https://github.com/kren-ai-lab/ruddy/blob/main/LICENSE).
