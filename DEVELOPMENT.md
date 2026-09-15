# Development Guide

## Prerequisites

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup

```bash
git clone https://github.com/kren-ai-lab/ruddy
cd ruddy
uv sync --all-extras
```

## Common Tasks

```bash
uv run task format    # sort imports + ruff format
uv run task lint      # ruff check (no fixes)
uv run task lint-fix  # ruff check --fix
uv run task test      # pytest -q
uv run task test-v    # pytest -v
uv run task test-cov  # pytest + HTML coverage report
uv run task ty        # ty check
uv run task pyrefly   # pyrefly check
```

## Running the CLI

```bash
uv run ruddy --version
uv run ruddy --help
```

## Scientific Gates

In addition to the regular test suite, Ruddy ships two dedicated gates under
`tools/`:

```bash
# Full scientific-freeze gate: compiles the package and tests, then runs the
# phase10 regression suite followed by the full test suite.
python tools/run_scientific_freeze_gate.py

# Executes every visualization notebook under examples/notebooks/ to confirm
# they still run against the current scientific core. Default mode runs each
# notebook's code cells in an isolated Python process (MPLBACKEND=Agg); pass
# --mode jupyter for a full nbconvert execution (--inplace to persist outputs).
python tools/run_notebook_gate.py
python tools/run_notebook_gate.py --mode jupyter --inplace
```

Run the scientific freeze gate before and after any change to a module under
`ruddy/` other than the CLI or non-scientific plumbing — see `AGENTS.md` for
the scientific-freeze rule.

## Project Structure

See `AGENTS.md` for the full package layout under `ruddy/`.

```text
ruddy/       # Flat package layout (migrated from src/ruddy/)
tests/       # Mirrors the source tree
tools/       # Scientific freeze and notebook validation gates
examples/    # Scripts and visualization notebooks
docs/        # Technical documentation
```
