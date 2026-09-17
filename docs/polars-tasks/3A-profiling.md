# Task 3A — `ruddy/profiling` returns Polars tables

Branch: `feat/polars-migration`. Prior tasks: `TabularDataset` stores Polars
(`dataset.frame`, `dataset.observation_id_tuple`, `dataset.schema`); pandas
adapters `to_frame()`, `select()`, `observation_ids` still exist and are
being retired. `write_table` and `validate_result_table` already accept
Polars. Read `AGENTS.md` and the "Semánticas" section of
`docs/POLARS_MIGRATION_PLAN.md` first.

You have no terminal. Do not run commands; the reviewer runs lint and tests.
No new modules or abstractions. Eager Polars only.

## Files to change

- `ruddy/profiling/columns.py`, `missingness.py`, `overview.py`
- `ruddy/univariate/numeric.py`, `categorical.py`, `distributions.py`,
  `outliers.py`: **only** the boundary where they consume the profiling
  `columns` table (see §4). Nothing else in univariate changes in this task.
- `tests/profiling/test_profiling.py`, `tests/profiling/test_missingness.py`
- `tests/integration/test_analysis_pipeline.py` line ~54 (one
  `assert_frame_equal` → `polars.testing.assert_frame_equal`)

## 1. `columns.py` — `profile_columns(dataset) -> pl.DataFrame`

Iterate `dataset.frame.columns`; for each column `series =
dataset.frame.get_column(column)`, `role/kind` from the dataset as today.

- `n_missing`: missing means `null`, **and** `NaN` for float columns
  (Ruddy counts NaN as missing; Polars does not). Use
  `series.null_count()` plus `series.is_nan().sum()` when
  `series.dtype.is_float()`.
- `n_unique`: `series.drop_nulls()` (and drop NaN for floats) then
  `n_unique()`. Wrap in `try/except (pl.exceptions.PolarsError, TypeError)`
  → `None`, matching the current `_safe_unique_count` contract for
  un-hashable kinds (`List`, `Struct`, `Object`).
- `finite_count`/`non_finite_count` for NUMERIC only: present values
  (non-null, non-NaN) cast to `Float64`, then `is_infinite()`. Non-numeric →
  `None, None`.
- `dtype`: `str(series.dtype)`.
- Build the table with `pl.DataFrame(rows, schema=COLUMN_PROFILE_SCHEMA)` where
  you add a module constant

  ```python
  COLUMN_PROFILE_SCHEMA: dict[str, pl.DataType] = {
      "column": pl.String, "dtype": pl.String, "role": pl.String, "data_kind": pl.String,
      "excluded": pl.Boolean, "analysis_eligible": pl.Boolean,
      "n_total": pl.Int64, "n_present": pl.Int64, "n_missing": pl.Int64,
      "missing_fraction": pl.Float64, "n_unique": pl.Int64, "unique_fraction": pl.Float64,
      "finite_count": pl.Int64, "non_finite_count": pl.Int64,
      "is_all_missing": pl.Boolean, "is_constant": pl.Boolean,
  }
  ```
  and keep `COLUMN_PROFILE_COLUMNS = tuple(COLUMN_PROFILE_SCHEMA)`. `None`
  values become nulls. An empty dataset (no columns) yields an empty frame
  with this schema (`pl.DataFrame(schema=...)`).

## 2. `missingness.py`

- `summarize_missingness(columns: pl.DataFrame) -> pl.DataFrame`: same
  required-field check; build with `columns.select(...)` expressions
  (`pl.col("column").cast(pl.String)`, `(1.0 - pl.col("missing_fraction")).alias("fraction_present")`,
  …) in `MISSINGNESS_COLUMNS` order. Keep `finite_count`/`non_finite_count`
  nullable ints.
- `pairwise_completeness(dataset, *, columns=None, max_columns=200) -> pl.DataFrame`:
  `selected = columns or tuple(dataset.frame.columns)`; presence mask per
  column as `pl.Series` of booleans (`is_not_null()`, and `& ~is_nan()` for
  floats) computed once into a dict; then the same
  `combinations_with_replacement` loop with `(mask_x & mask_y).sum()`. Declare
  a `PAIRWISE_COMPLETENESS_SCHEMA` dict like §1 and build with it.
- `summarize_missingness_patterns(dataset, *, max_patterns=20) -> pl.DataFrame`:
  compute the boolean missing frame (`null` or float `NaN`) as a Polars frame,
  then `iter_rows()` to build the same `counts` dict and rows; same ordering
  and JSON encoding of patterns; `MISSINGNESS_PATTERN_SCHEMA` dict.
- The `n_total` for an empty dataset stays `dataset.frame.height`.

## 3. `overview.py`

- `_identifier_summary`: use `ids = dataset.observation_id_tuple`; the IDs are
  already validated unique, so `n_unique = len(ids)`, duplicates `0`.
  `"source"` value: `"column"` if `dataset.id_column` else
  `dataset.provenance["id_source"]` (`"argument"` or `"generated"`).
- `summarize_overview(dataset, columns: pl.DataFrame)`: `frame =
  dataset.frame`; missing mask as in §2; `n_missing_cells = mask.sum_horizontal().sum()`
  or equivalent; `rows_with_missing = mask.select(pl.any_horizontal(pl.all()))`
  (for a frame with zero columns the result is all-False of length `height`);
  duplicates: `frame.is_duplicated()` (all rows in duplicate groups) and
  `frame.is_duplicated() & ~frame.is_first_distinct()` (extra rows). For a
  frame with zero columns both are zero.
  Role/kind counts, quality counters and `numeric_non_finite` via Polars
  filters on `columns` (`fill_null(0)` before summing). `n_records =
  frame.height`, `n_columns = frame.width`.
- `ProfilingResult` fields become `pl.DataFrame`. Update the provenance
  parameter `"missingness_policy"` to `"null_or_nan_missing_values"`.
- `profile_dataset` unchanged otherwise.

## 4. Univariate boundary (temporary adapters)

Univariate still runs on pandas. Where it consumes the profiling columns
table, convert once at the boundary, each with the comment
`# ponytail: temporary pandas adapter, removed in task 3B`:

- `distributions.py`: after `profiling = profile_dataset(...)`, use
  `columns = profiling.columns.to_pandas()` and pass that `columns` to the
  three `summarize_*` calls (check how they receive it today and keep their
  signatures).
- `outliers.py` lines ~141, ~495, ~561: `profile_columns(dataset).to_pandas()`
  when `columns is None`; if a caller passes `columns` that is a
  `pl.DataFrame`, convert it too (`columns.to_pandas() if isinstance(columns, pl.DataFrame) else columns`).
- `numeric.py`, `categorical.py`: only if they call `profile_columns`
  directly; otherwise untouched.

## 5. Tests

Rewrite `tests/profiling/*` assertions for Polars: replace
`table.set_index("column").loc[name, field]` with a small local helper
`_row(table, name) -> dict` = `table.filter(pl.col("column") == name).row(0, named=True)`;
`.empty` → `.height == 0`; `.iloc[0][field]` → `table.row(0, named=True)[field]`;
frame equality with `polars.testing.assert_frame_equal`. Keep every existing
behavioural assertion. Add:

- `profile_columns` on a Polars frame with a `Float64` column containing
  `None`, `float("nan")` and `inf`: `n_missing == 2`, `n_present == 1`,
  `finite_count == 0`, `non_finite_count == 1`.
- `profile_columns` output schema equals `COLUMN_PROFILE_SCHEMA` (dtypes), also
  on an empty `TabularDataset(pl.DataFrame())`.
- overview `identifier.source == "generated"` for a frame without ID column.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task test
```

542 tests plus additions must pass. If a test outside the listed files fails
and the cause is a pandas idiom on a profiling table (e.g. in `tests/cli`,
`tests/robustness`, `tests/univariate`), name the file and line and the idiom
in your summary instead of editing it. Finish with files touched, tests
added, anything not done.
