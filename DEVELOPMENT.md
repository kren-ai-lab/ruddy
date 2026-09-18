# Development Guide

## Prerequisites

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) (package manager)

## Dependencies

Core dependencies are managed via `uv`:
- **Polars** (`polars>=1.44,<2`) and **pyarrow** (`pyarrow>=25,<26`) form the core tabular data engine and I/O backend.
- **NumPy**, **SciPy**, **scikit-learn**, and **statsmodels** provide numeric and statistical modeling engines.
- **pandas** is accepted as an input format for `TabularDataset` and `FeatureMatrix` and is retained transitively via statsmodels.
- **Typer** and **Rich** power the CLI.
- Optional dependencies include `umap-learn` (`.[manifold]`) and the `examples` group for marimo/plotting libraries.

## Setup

```bash
git clone https://github.com/kren-ai-lab/ruddy
cd ruddy
uv sync --all-extras
# optional: dependencies for the examples
uv sync --all-extras --group examples
```

## Common Tasks

```bash
uv run task format    # sort imports + ruff format
uv run task lint      # ruff check (no fixes)
uv run task lint-fix  # ruff check --fix
uv run task test      # pytest -q
uv run task test-v    # pytest -v
uv run task test-cov  # pytest + HTML coverage report
uv run task pyrefly   # pyrefly check
```

## Running the CLI

```bash
uv run ruddy --version
uv run ruddy --help
```

## Examples

The examples under `examples/` are marimo notebooks stored as plain Python files
(requires the `examples` group):

```bash
bash examples/run_ci_examples.sh                         # run them all, as CI does
uv run marimo edit examples/01_profiling_univariate.py   # open one interactively
```

## Project Structure

See `AGENTS.md` for the full package layout under `ruddy/`.

```text
ruddy/       # Flat package layout
tests/       # One directory per ruddy/ package + integration, robustness, parity
examples/    # marimo examples (plain .py) and demo data
docs/        # Technical documentation
```
