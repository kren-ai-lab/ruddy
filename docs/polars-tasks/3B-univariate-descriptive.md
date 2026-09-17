# Task 3B — Univariate descriptive tables in Polars

Branch: `feat/polars-migration`. State: `TabularDataset.frame` is Polars;
`profile_columns()` returns a Polars table with `COLUMN_PROFILE_SCHEMA`;
`write_table`/`validate_result_table` accept Polars. Read `AGENTS.md` and the
"Semánticas" section of `docs/POLARS_MIGRATION_PLAN.md` first. Look at
`ruddy/profiling/columns.py` for the schema-dict + `pl.DataFrame(rows,
schema=...)` pattern to follow.

You have no terminal. Do not run commands; the reviewer runs lint and tests.
No new modules or abstractions. Keep all numerics in NumPy/SciPy exactly as
they are; this task only changes how columns are read and how tables are
assembled.

## Files to change

- `ruddy/univariate/numeric.py`, `categorical.py`, `distributions.py`
- `ruddy/univariate/outliers.py`: **only** the temporary adapter lines marked
  `removed in task 3B` — see §5.
- `ruddy/univariate/__init__.py` if exports change (they should not).
- `tests/univariate/test_univariate.py`,
  `tests/parity/test_univariate_reference_parity.py`,
  `tests/cli/test_descriptive_cli.py`,
  `tests/integration/test_analysis_pipeline.py` and
  `tests/robustness/test_pathological_tabular.py`,
  `tests/robustness/test_cross_surface_parity.py`: only the assertions that
  touch `numeric_statistics`, `categorical_statistics`,
  `categorical_frequencies`, `datetime_statistics` or `UnivariateResult`.

`diagnostics.py`, `outliers.py` (beyond §5) and `statistics/` are out of
scope: task 3C.

## 1. Reading columns

Every `summarize_*` takes `columns: pl.DataFrame` (the profiling table) and
selects eligible rows with Polars, replacing `columns.apply(_eligible_x,
axis=1)` + `iterrows()`:

```python
selected = columns.filter(pl.col("analysis_eligible") & <kind/role predicate>)
for profile in selected.iter_rows(named=True):
    column = profile["column"]; role = profile["role"]
```

Predicates, unchanged in meaning:
- numeric: `role != "factor"` and `data_kind == "numeric"`
- categorical: `role == "factor"` or `data_kind in ("categorical", "boolean")`
- datetime: `role != "factor"` and `data_kind == "datetime"`

Delete the `_eligible_numeric/_eligible_categorical/_eligible_datetime`
`pd.Series` helpers in these three modules. (`outliers.py` and
`diagnostics.py` keep theirs until 3C; `bivariate/comparisons.py` imports
`diagnostics._eligible_numeric`, which is untouched.)

Source values come from `dataset.frame.get_column(column)` (a `pl.Series`).
Remove `dataset.to_frame()` from these modules.

## 2. `numeric.py`

- `_finite_numeric_values(series: pl.Series) -> np.ndarray`:
  `values = series.drop_nulls().cast(pl.Float64).to_numpy()`, then
  `values[np.isfinite(values)]` (this also drops NaN, as today).
- `n_missing` for the row = `null_count()` plus NaN count for float dtypes
  (same rule as `profiling/columns.py`); `n_present = n_total - n_missing`.
  Check the current `_numeric_row` for where these come from and keep the
  same numbers.
- **Skewness/kurtosis** lose pandas: use
  `scipy.stats.skew(values, bias=False)` and
  `scipy.stats.kurtosis(values, bias=False)` (Fisher, excess). Keep the
  existing guards (`not is_constant and n_finite >= 3` / `>= 4`) and
  `_safe_float`. Add to the module docstring one sentence noting the
  estimator (adjusted Fisher–Pearson G1 / unbiased excess G2, the same
  formulas pandas used).
- Table: build `NUMERIC_STATISTICS_SCHEMA(quantiles) -> dict[str, pl.DataType]`
  from `numeric_statistics_columns(quantiles)`: `column`, `role`, `status`,
  `reason` → `pl.String`; `n_*`, `zero_count` → `pl.Int64`; everything else →
  `pl.Float64`. Return `pl.DataFrame(rows, schema=...)`.

## 3. `categorical.py`

- `_category_label`: replace the `pd.Timestamp` branch with
  `datetime.datetime`/`datetime.date` → `.isoformat()` (Polars yields Python
  datetimes). Keep the rest. This helper is imported by `bivariate/*` and
  `analysis/intervals.py`; keep its name and signature.
- `_profile(..., series: pl.Series, ...)`: `present = series.drop_nulls()`
  (and drop NaN for float dtypes); `labels = [_category_label(v) for v in present.to_list()]`;
  counts via `collections.Counter(labels)`; ordering `(-count, level)` as
  today. Everything else unchanged.
- Schemas: `CATEGORICAL_STATISTICS_SCHEMA` (`mode` nullable String,
  `entropy_base` Int64, `frequencies_truncated` Boolean, fractions Float64,
  counts Int64) and `CATEGORICAL_FREQUENCIES_SCHEMA` (`column`, `role`,
  `level` String; `count`, `rank` Int64; `fraction` Float64). Return two
  Polars frames.

## 4. `distributions.py`

- `summarize_datetime_statistics(dataset, columns: pl.DataFrame) -> pl.DataFrame`:
  `present = series.drop_nulls()`; `min()/max()` give Python datetimes;
  `range_seconds = (maximum - minimum).total_seconds()`; constant check
  `present.n_unique() == 1`. Schema: `min`/`max` → `pl.Datetime("us")`
  (cast the series to that unit first if it is `pl.Date` or another unit:
  `present.cast(pl.Datetime("us"))`), `range_seconds` Float64, the rest as
  the numeric table.
- `UnivariateTables`/`UnivariateResult` fields become `pl.DataFrame`.
- `summarize_univariate`/`analyze_univariate`: drop the adapters and pass
  `profiling.columns` straight through.

## 5. `outliers.py` adapters

The three `# ponytail: temporary pandas adapter, removed in task 3B` blocks
stay **as they are** (outliers still runs on pandas until 3C), but update the
comment text to `removed in task 3C`. Do not change anything else there.

## 6. Tests

Translate assertions to Polars using the helpers already used in
`tests/profiling/test_profiling.py` (`_row(table, name)` via
`filter(...).row(0, named=True)`, `.height`, `row[...] is None` instead of
`pd.isna`). Datetime `min`/`max` compare with `datetime.datetime(2026, 1, 1)`.
Frame equality with `polars.testing.assert_frame_equal`.

`tests/parity/test_univariate_reference_parity.py`: keep the frozen JSON as is.
Compare per column: `table.get_column(c).to_list()` against the reference
list, treating `"NaN"` in the reference and `None` in the table as equal
missing, and numbers with `np.allclose(rtol=1e-12, atol=1e-12, equal_nan=True)`
after mapping missing to NaN. Filter rows with
`table.filter(pl.col("column").is_in(expected_columns))`; keep the row order.

Add:
- a near-constant numeric column, e.g. `[1.0, 1.0, 1.0, 1.0 + 1e-13]`:
  assert `status == "ok"` and that `skewness` and `kurtosis` are finite
  floats (pins the SciPy estimator on the edge pandas used to zero out).
- `numeric_statistics` schema dtypes: `n_finite` is `pl.Int64`, `mean` is
  `pl.Float64`, `status` is `pl.String`, for an empty result too
  (dataset with no numeric columns → `height == 0`, same schema).
- `categorical_frequencies` for a Boolean Polars column yields levels
  `"false"`/`"true"`.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task test
```

545 tests plus additions must pass. If the parity test fails on skewness or
kurtosis, do not loosen tolerances and do not edit the reference: report the
column, the observed and expected values, and the input values. If a test
outside the listed files fails, name file, line and idiom in your summary.
Finish with files touched, tests added, anything not done.
