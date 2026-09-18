# Development Guide

## Prerequisites

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) (package manager)

## Dependencies

Core dependencies are managed via `uv`:
- **Polars** (`polars>=1.44,<2`) and **pyarrow** (`pyarrow>=25,<26`) form the core tabular data engine and I/O backend.
- **NumPy**, **SciPy**, **scikit-learn**, and **statsmodels** provide numeric and statistical modeling engines.
- **pandas** remains a direct dependency for accepted inputs to `TabularDataset` and `FeatureMatrix` and the explicit statsmodels/Patsy model boundaries.
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

The package uses a flat layout:

```text
ruddy/       # Flat package layout
tests/       # One directory per ruddy/ package + integration, robustness, parity
examples/    # marimo examples (plain .py) and demo data
docs/        # Technical documentation
```

## Making changes

Read [AGENTS.md](AGENTS.md) for the scientific invariants. Until the first
release, the set of scientific methods is closed; fixes and refactors are
welcome. A change to numerical results needs a regression test that pins the
new behavior.

Keep scientific calculations in `ruddy/` and plotting in `examples/`. The core
must not import Matplotlib or Plotly. Standalone APIs, the unified analysis and
CLI exports must agree for the same inputs and options.

Polars is the tabular contract and NumPy/SciPy handle numerical matrices.
Formula models use explicit pandas conversions for statsmodels/Patsy. When
changing model preparation, preserve observation IDs, categorical coding,
complete-case exclusions and the public Polars result schemas.

## Choosing checks

Run `uv run task lint`, `uv run task pyrefly` and `uv run task test` before
submitting code changes. Use the suite nearest the affected behavior while
iterating:

| Location | What it checks |
| --- | --- |
| `tests/<package>/` | Public behavior of the corresponding package |
| `tests/integration/` | Unified analysis and composition of blocks |
| `tests/parity/` | Numerical reference agreement and supported input formats |
| `tests/robustness/` | Missing/constant data, rank and dimensionality guards, sparse inputs, alignment and reproducibility |
| `tests/cli/` | Command behavior and exported artifacts |

For scientific changes, cover the relevant degenerate case as well as an
estimable case. Verify sample counts/exclusions, statuses, nulls, schemas and
provenance where affected. Stochastic checks should use explicit seeds.

When changing examples or the API they consume, run
`bash examples/run_ci_examples.sh`. This regenerates demo data and executes all
13 notebooks. Review generated-data diffs before committing them.

For documentation-only changes, verify local links, referenced public names,
CLI options and runnable snippets. Update the guide for the affected behavior
and keep it about the current API. The [documentation index](docs/README.md)
is the user entry point; examples are catalogued only in
[examples/README.md](examples/README.md).
