# Task 3C — Outliers, distribution diagnostics and multiple testing in Polars

Branch: `feat/polars-migration`. State: `TabularDataset.frame` is Polars;
profiling and univariate descriptive tables (`numeric.py`, `categorical.py`,
`distributions.py`) return Polars with declared schemas — copy their
patterns (`filter(...)` + `iter_rows(named=True)`, schema dicts,
`pl.DataFrame(rows, schema=...)`, `drop_nulls()` + NaN filter for floats).
Read `AGENTS.md` and the "Semánticas" section of
`docs/POLARS_MIGRATION_PLAN.md` first.

You have no terminal. Do not run commands; the reviewer runs lint and tests.
No new modules or abstractions. Keep every numeric computation in
NumPy/SciPy exactly as it is.

## Files to change

- `ruddy/univariate/outliers.py`, `ruddy/univariate/diagnostics.py`
- `ruddy/statistics/multiple_testing.py`
- `ruddy/cli/commands/outliers.py` (only `.empty` → `.height == 0`, and
  `shape[0]` → `height`) and `ruddy/cli/commands/inference.py` (only if a
  pandas idiom on `normality`/`dispersion` remains; `len()` works on Polars).
- `tests/univariate/test_outliers.py`, `test_outlier_degenerate_cases.py`,
  `test_distribution_diagnostics.py`, `tests/statistics/test_multiple_testing.py`,
  `tests/cli/test_outliers_cli.py`; plus only the assertions on outlier or
  diagnostics tables in `tests/integration/test_analysis_pipeline.py`,
  `tests/robustness/test_pathological_tabular.py`,
  `tests/robustness/test_cross_surface_parity.py`.

`bivariate/` is out of scope (task 3D); it keeps calling
`apply_multiple_testing` with pandas, so that function must accept both.

## 1. `statistics/multiple_testing.py`

- `adjust_pvalues` unchanged.
- `family_sizes(table)` and `apply_multiple_testing(table, ...)` accept
  `pl.DataFrame | pd.DataFrame`. Keep the pandas bodies as they are under
  `# ponytail: temporary pandas adapter, removed in task 3D`, and add a
  Polars path with the same semantics:
  - required-column check and `correction` consistency check as today
    (`table.get_column(correction_column).cast(pl.String) == resolved.value`
    must be all true; empty table returns early after adding the columns);
  - `family_id` must be non-null and non-blank after `str.strip_chars()`;
  - compute with NumPy: `p = table.get_column(p_column).cast(pl.Float64).to_numpy()`
    (nulls → NaN via `fill_null(float("nan"))` first), `status`, `family`
    arrays; `q = np.full(n, np.nan)`; `sizes = np.zeros(n, dtype=np.int64)`;
    loop over `dict.fromkeys(family)` in first-seen order; inferential mask
    = family & status == "ok" & isfinite(p); `sizes[family_mask] = k`;
    `q[inferential] = adjust_pvalues(p[inferential], resolved)`;
  - return `table.with_columns(pl.Series(q_column, q, dtype=pl.Float64).fill_nan(None), pl.Series(family_size_column, sizes, dtype=pl.Int64))`.
    If `q_column` already exists it is replaced. NaN becomes null: q-values
    are missing, not NaN, in Polars tables.
  - `family_sizes` Polars path: `table.filter(pl.col(status_column) == ok_status).get_column(family_column).cast(pl.String)`,
    then counts in first-seen order.

## 2. `outliers.py`

- Delete the `_eligible_numeric(profile: pd.Series)` helper; select with
  `columns.filter(pl.col("analysis_eligible") & (pl.col("role") != "factor") & (pl.col("data_kind") == "numeric"))`
  and `iter_rows(named=True)`. `columns` parameters become
  `pl.DataFrame | None`; when `None`, `profile_columns(dataset)`. Remove the
  three `removed in task 3C` adapter blocks and `dataset.to_frame()`.
- `_finite_values_with_positions(series: pl.Series)`: `missing` mask =
  `is_null()` or (float and `is_nan()`) as a NumPy bool array; `numeric =
  series.cast(pl.Float64).fill_null(float("nan")).to_numpy()` (for non-float
  ints this cast is exact); the rest unchanged.
- `observation_ids`: use `dataset.observation_id_tuple` (a tuple; indexing
  `observation_ids[position]` still works). Type hints
  `tuple[ObservationID, ...]`.
- Schemas: `NUMERIC_QUALITY_SCHEMA`, `OUTLIER_SUMMARY_SCHEMA`,
  `OUTLIER_FLAG_SCHEMA_BASE` (everything except `observation_id`) built from
  the existing `*_COLUMNS` tuples: `column/role/method/criterion/direction/status/reason`
  → String; `n_*`, `*_count`, `source_row_index` → Int64; fractions,
  `threshold`, `value`, `score`, `iqr`, `mad`, `center`, `scale`, bounds →
  Float64; `has_*`, `zero_*` → Boolean. Check the current row dicts for the
  exact key list; any numeric key that can be `None` stays nullable
  Float64/Int64 (Polars handles that).
- `observation_id` in the flags table keeps the ID type: build the flags
  frame with `pl.DataFrame(rows, schema_overrides=OUTLIER_FLAG_SCHEMA_BASE)`
  so `observation_id` is inferred, and for zero flag rows use
  `pl.DataFrame(schema={**OUTLIER_FLAG_SCHEMA_BASE, "observation_id": pl.Series(dataset.observation_id_tuple).dtype})`
  reordered to `OUTLIER_FLAG_COLUMNS`. A helper of a few lines inside
  `outliers.py` is fine; no new module.
- `OutlierResult` fields become `pl.DataFrame`.

## 3. `diagnostics.py`

- `_finite(series: pl.Series)`: `series.drop_nulls().cast(pl.Float64).to_numpy()`
  then `isfinite` filter.
- Normality: `frame = dataset.frame`; `x = _finite(frame.get_column(column))`;
  table with `NORMALITY_SCHEMA`; `return table if table.height == 0 else apply_multiple_testing(table, correction)`.
- Dispersion: replace the pandas `pair` block with
  `pair = frame.select(response, group).drop_nulls(group)`; values
  `pair.get_column(response).cast(pl.Float64).fill_null(float("nan")).to_numpy()`;
  keep finite rows via a NumPy mask applied to both arrays; labels
  `[_category_label(v) for v in pair.get_column(group).to_list()]` filtered
  by the same mask; `levels = sorted(set(labels))`; samples per level with
  NumPy. Same `sizes`, statuses and reasons. `DISPERSION_SCHEMA`.
  `pd.to_numeric(..., errors="coerce")` disappears: the response is already a
  numeric column by eligibility, so a plain cast is exact.
- `analyze_distribution_diagnostics`: the empty-dispersion fallback becomes
  `pl.DataFrame(schema=DISPERSION_SCHEMA)`.
- `DistributionDiagnosticsResult` fields become `pl.DataFrame`.
- `_eligible_numeric(dataset)` and `_eligible_groups(dataset)` stay as they
  are (imported by `bivariate/comparisons.py`).

## 4. Tests

Translate as in `tests/univariate/test_univariate.py`: row lookup via
`filter(...).row(0, named=True)`, `.height == 0` for emptiness, `is None`
for missing, `polars.testing.assert_frame_equal` for frame equality
(determinism tests with `check_exact=True`). `test_multiple_testing.py`: keep
the pandas test and add the same case on a Polars table asserting
`q_value` nulls on non-ok rows and `family_size` values.

Add:
- flags `observation_id` dtype follows the dataset IDs: string IDs →
  `pl.String`; generated integer IDs → `pl.Int64`; a dataset producing no
  flags still has an `observation_id` column with the ID dtype.
- Polars `apply_multiple_testing` rejects a blank `family_id` and an
  inconsistent `correction` value with `ValueError`.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task test
```

548 tests plus additions must pass. If a test outside the listed files fails,
name file, line and idiom in your summary. Finish with files touched, tests
added, anything not done.
