# Task 3D — Bivariate core in Polars: correlations, comparisons, associations, contingency

Branch: `feat/polars-migration`. State: `TabularDataset.frame` is Polars;
profiling, univariate, outliers, diagnostics return Polars with declared
schemas; `apply_multiple_testing` accepts Polars (q-values null when
missing). Copy the patterns from `ruddy/univariate/numeric.py`,
`ruddy/univariate/diagnostics.py` (dispersion block) and
`ruddy/univariate/outliers.py`. Read `AGENTS.md` and the "Semánticas"
section of `docs/POLARS_MIGRATION_PLAN.md` first.

You have no terminal. Do not run commands; the reviewer runs lint and tests.
No new modules or abstractions. Every statistic stays in NumPy/SciPy as it
is; only column access and table assembly change.

## Files to change

- `ruddy/bivariate/correlations.py`, `comparisons.py`, `associations.py`,
  `contingency.py`, `analysis.py`
- `ruddy/cli/commands/bivariate.py`, `ruddy/cli/commands/inference.py`: only
  pandas idioms on these results (`len()` is fine on Polars).
- Tests: `tests/bivariate/test_correlations.py`, `test_comparisons.py`,
  `test_associations.py`, `test_contingency.py`, `test_analysis.py`,
  `tests/parity/test_bivariate_reference_parity.py`; and only assertions on
  these tables in `tests/integration/test_analysis_pipeline.py`,
  `tests/robustness/test_cross_surface_parity.py`,
  `tests/robustness/test_inference_torture.py`,
  `tests/robustness/test_pathological_tabular.py`, `tests/cli/*`.

Out of scope (task 3E): `dependence.py`, `posthoc.py`, `groups.py`.
`groups.py` calls `summarize_categorical_associations` and
`comparisons` helpers: **keep every public and `_private` function name and
signature you touch**, and make sure any helper that `groups.py`,
`posthoc.py` or `dependence.py` imports from these modules still works when
given a Polars `frame` (grep for `from ruddy.bivariate.` in `ruddy/` before
changing a helper). If one of those callers passes a pandas frame to a
helper you changed, add at the top of that helper
`# ponytail: temporary pandas adapter, removed in task 3E` +
`if isinstance(frame, pd.DataFrame): frame = pl.from_pandas(frame)` and
list it in your summary.

## Common rules

- Source access: `frame = dataset.frame`; a column is
  `frame.get_column(name)`.
- Finite numeric values of a column: `series.cast(pl.Float64).fill_null(float("nan")).to_numpy()`
  then `np.isfinite` masks (pairwise: both finite).
- Category labels: `[_category_label(v) for v in series.to_list()]` on the
  non-missing rows; missing = null, or NaN for floats.
- Every table gets a `<NAME>_SCHEMA: dict[str, pl.DataType]` next to the
  existing `<NAME>_COLUMNS` tuple (make the tuple `tuple(SCHEMA)`); build with
  `pl.DataFrame(rows, schema=SCHEMA)`; empty results are
  `pl.DataFrame(schema=SCHEMA)`. Types: names/labels/status/reason/method/
  family_id/correction → String; counts, df, ranks, `n_*` → Int64;
  statistics, p/q, effect sizes, fractions, bounds → Float64; flags →
  Boolean. Nullable where the row dicts can hold `None`.
- `return table if table.height == 0 else apply_multiple_testing(table, correction)`.

## Module notes

- `correlations.py`: `_pairwise_finite(left: pl.Series, right: pl.Series)`.
- `comparisons.py`: the group-splitting helper around line 291 (builds
  `normalized` labels and per-group arrays with `.loc`) becomes: labels list
  from `_category_label` over the group column, `present` mask = not
  missing; `labels = tuple(sorted(set(labels_present)))`; per-label arrays via
  NumPy boolean masks over the feature values. `_finite_values(series: pl.Series)`.
  Line ~357 (`raw_present = frame[group].dropna().map(...)`) same idea.
  Keep `_eligible_numeric` import from `univariate.diagnostics` as is.
- `associations.py`: `_contingency(frame: pl.DataFrame, x, y)`:
  `pair = frame.select(x, y).drop_nulls()` (and drop float NaN if either is
  float), labels via `_category_label`, then the existing NumPy contingency
  construction. Return types unchanged.
- `contingency.py`: same for the crosstab construction; `summary` and
  `cells` become Polars with `SUMMARY_SCHEMA`/`CELL_SCHEMA`.
- `analysis.py`: `BivariateResult` fields `pl.DataFrame`;
  `dataset.n_columns` → `dataset.frame.width`.

## Tests

Translate as in `tests/univariate/`: `filter(...).row(0, named=True)`,
`.height`, `is None` for missing, `polars.testing.assert_frame_equal`.
`tests/parity/test_bivariate_reference_parity.py`: keep the JSON reference;
build the `(method, column_x, column_y) -> row` dict from
`table.iter_rows(named=True)` and compare values with the existing
`_assert_value` (a `None` counts as the `"NaN"` reference token).

Add: `summarize_correlations` on a Polars-native dataset with an all-null
`Float64` column reports the same `status/reason` as the pandas-era test for
an all-NaN column; the correlations schema has `q_value` as `pl.Float64`
and `family_size` as `pl.Int64` also when the table is empty.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task test
```

554 tests plus additions must pass. If the bivariate parity test fails, do
not loosen tolerances or edit the reference: report method, columns,
observed and expected. If a test outside the listed files fails, name file,
line and idiom. Finish with files touched, tests added, adapters added,
anything not done.
