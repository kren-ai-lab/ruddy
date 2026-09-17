# Development Guide

## Prerequisites

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup

```bash
git clone https://github.com/kren-ai-lab/ruddy
cd ruddy
uv sync --all-extras
# optional: dependencies for the example notebooks
uv sync --all-extras --group notebooks
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

## Notebook Gate

`tools/run_notebook_gate.py` executes every notebook under `examples/notebooks/`
against the current scientific core (requires the `notebooks` group). Default
mode runs each notebook's code cells in an isolated Python process
(`MPLBACKEND=Agg`); `--mode jupyter` performs a full nbconvert execution
(`--inplace` persists outputs).

```bash
uv run python tools/run_notebook_gate.py
uv run python tools/run_notebook_gate.py --mode jupyter --inplace
```

Run the full test suite before and after any change to a module under `ruddy/`
other than the CLI or non-scientific plumbing — see `AGENTS.md` for the
scientific-freeze rule.

## Project Structure

See `AGENTS.md` for the full package layout under `ruddy/`.

```text
ruddy/       # Flat package layout
tests/       # One directory per ruddy/ package + integration, robustness, parity, examples
tools/       # Notebook validation gate
examples/    # Scripts and visualization notebooks
docs/        # Technical documentation
```
