# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

**Ruddy** is a domain-agnostic Python library for statistical exploratory data
analysis (EDA) of tabular datasets and numerical feature spaces (embeddings,
descriptor matrices, structural encodings, or any other numerical
representation with identifiable observations). It sits alongside three
sibling libraries from the same lab, each with a distinct scope:

- **Roxy** — classical protein sequence descriptors
- **Sylphy** — sequence encoders, embeddings and dimensionality reductions
- **Biosieve** — dataset deduplication and leakage-aware partitioning

Ruddy does **not** duplicate any of that functionality and does **not**
depend on any of the sibling libraries. It only consumes tabular data and
numerical feature matrices through its own `TabularDataset` / `FeatureMatrix`
contracts — a Roxy descriptor table, a Sylphy embedding, or any other numeric
matrix is just an input to Ruddy, never a dependency.

This project uses `uv` and `taskipy` — see `DEVELOPMENT.md` for setup and
task commands.

## Package Layout

The package is a **flat layout** at `ruddy/`. `tests/` mirrors it one directory
per package, plus the cross-cutting suites `integration/`, `robustness/`,
`parity/` and `examples/`.

## Development Workflow

Package management is `uv`; task running is `taskipy`. See `DEVELOPMENT.md`
for the full command list (`uv sync --all-extras`, `uv run task
format|lint|lint-fix|test|test-v|test-cov|pyrefly`).

## Scientific Invariants (read before touching any scientific module)

These are the most important rules for an agent working in this repository.
They come from the "Design principles" section of `README.md`.

- **No silent coercion, row deletion, scaling, or dimensionality reduction.**
  Anything that changes what data is used or how it's transformed must be
  explicit and recorded, never inferred automatically.
- **No automatic method switching.** Diagnostics (normality, variance
  homogeneity, etc.) never silently change the inferential test the user
  requested.
- **Explicit degenerate/skipped states, not silent NaNs.** Rank-deficient or
  otherwise ill-posed problems are surfaced as structured `degenerate` or
  `skipped` results with a `reason`, not masked.
- **Outliers/anomalies are flagged, never removed or modified.**
- **Identity before row position.** Data, annotations, and representation
  matrices are aligned by observation IDs, not row order.
- **Structured provenance.** Parameters, input summaries, seeds, and the
  Ruddy version are retained in result contracts.

**Feature freeze until the first release.** The set of scientific methods is
closed: do not add new analyses until the release. The code itself is not
frozen — refactors, cleanups and fixes are welcome as long as the invariants
above hold and the test suite passes. A change that alters numerical results
needs a test that pins the new behavior.

## Core vs. Visualization

The scientific core (`ruddy/`) has no dependency on plotting libraries
(Matplotlib, Plotly, etc.) and must never import one. It returns structured,
traceable result objects only. All visualization lives externally, in the
example notebooks under `examples/notebooks/`, which consume those result
objects — they never recompute statistics themselves.
