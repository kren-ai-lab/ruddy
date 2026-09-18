# Task 6A — Remove the temporary pandas adapters; Polars end to end

Branch: `feat/polars-migration`. State: every analysis returns Polars.
pandas survives only in (a) the four explicit model boundaries
(`factorial/{models,marginal_means,mixed_effects}.py`, `multivariate/manova.py`,
marked `# pandas boundary: ...`), which **stay untouched**, and (b) the
temporary adapters marked `# ponytail: temporary pandas adapter, removed in
phase 5` plus the CLI/IO reading path. This task removes (b). Read
`AGENTS.md` and the plan (`docs/POLARS_MIGRATION_PLAN.md`) first.

You have no terminal. Do not run commands; the reviewer runs lint, pyrefly
and tests. No new modules or abstractions.

## Target public contract (after this task)

`TabularDataset`:
- keeps: `frame: pl.DataFrame`, `observation_ids: tuple[ObservationID, ...]`
  (**now the tuple**; `observation_id_tuple` is deleted), `id_column`,
  `schema`, `provenance`, `role_of`, `kind_of`, `columns_with_role`,
  `columns_with_kind`, `__repr__`.
- deleted: `to_frame()`, `select()`, `n_observations`, `n_columns`,
  `columns`, `__len__`. Callers use `dataset.frame.height`,
  `dataset.frame.width`, `dataset.frame.columns`, `dataset.frame.select(...)`.
- input: `pl.DataFrame | pd.DataFrame` (pandas conversion stays; it is a
  user-facing convenience, not an adapter).

`FeatureMatrix`: `observation_ids` returns the tuple; `observation_id_tuple`
deleted. `metadata` stays Polars.

`AlignedAnnotations`: `frame` stays; `to_frame()` deleted.

`align_annotations`: annotations must be `pl.DataFrame` **or**
`pd.DataFrame` **with `id_column`**. The pandas-index-as-identity path is
removed: a pandas frame without `id_column` raises
`ValueError("Annotations require id_column; a pandas index is not an identity. Pass frame.reset_index() with the ID as a column.")`.
`validate_observation_ids` drops `pd.NA` handling only if pandas is no longer
imported in `validation.py`; if a pandas input path remains there, keep it.

`validate_result_table` and `apply_multiple_testing`/`family_sizes`: Polars
only; delete the pandas branches and their overloads.

`core/io.py`: `read_table(path) -> pl.DataFrame` (`pl.read_csv(...,
separator=...)`, `pl.read_parquet`); `write_table(frame: pl.DataFrame, path)`
without `index`/`index_label`. Reading a Parquet written by pandas with an
index column keeps that column as data (document in the docstring).
CSV reading: use `infer_schema_length=None`, `try_parse_dates=True`, and
`null_values=["", "NA", "NaN", "null"]` **only if** the current pandas tests
in `tests/cli` and `tests/core` (if any) need them to pass; otherwise Polars
defaults. Report which you chose and why.

## Files to change

- `ruddy/data/dataset.py`, `feature_matrix.py`, `annotations.py`,
  `validation.py`, `ruddy/core/types.py` (drop pandas from aliases that no
  longer need it: `ObservationIDs` keeps `pd.Index | pd.Series` only if
  `FeatureMatrix` still accepts them — it does, via pandas input; keep),
  `ruddy/results/schemas.py`, `ruddy/statistics/multiple_testing.py`,
  `ruddy/core/io.py`.
- Every consumer of a deleted member (grep `ruddy/` for `.to_frame()`,
  `dataset.select(`, `.observation_id_tuple`, `.n_observations`,
  `.n_columns`, `dataset.columns`, `len(dataset)`): ~80 sites across
  `analysis/`, `anomaly/`, `compositional/`, `factorial/`, `multivariate/`,
  `profiling/`, `projections/`, `representation/`, `univariate/`, `cli/`.
  Where a consumer used `dataset.columns` for membership, use
  `dataset.frame.columns`.
- CLI: `ruddy/cli/commands/descriptive.py` (`load_dataset`, `read_table`
  consumer), `projections.py`, `groups.py`, `_console.py`, and any
  `pd.DataFrame` construction left in `ruddy/cli/`.
- Tests: adapt everything that used the deleted members or pandas
  annotations-by-index (`tests/data/*`, `tests/cli/*`, `tests/core/*`,
  `tests/integration/*`, `tests/robustness/*`, `conftest.py` fixtures may
  keep building pandas inputs — that path stays supported).

## Acceptance (reviewer)

```bash
uv run task lint
uv run task pyrefly
uv run task test
grep -rn "ponytail: temporary" ruddy      # must print nothing
grep -rln "import pandas" ruddy | sort   # expected: core/types.py, data/dataset.py, data/feature_matrix.py, data/validation.py (input conversion only), factorial/{models,marginal_means,mixed_effects}.py, multivariate/manova.py
```

If a pandas import remains anywhere else, justify it in the summary. All
tests plus additions must pass; pyrefly stays at one error (optional `umap`
import). If you cannot finish a consumer group, leave the codebase
consistent (tests passing) and list precisely what is left. Finish with
files touched, tests changed, the CSV-reading choice, anything not done.
