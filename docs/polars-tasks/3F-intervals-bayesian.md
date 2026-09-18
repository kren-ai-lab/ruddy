# Task 3F — Confidence intervals and Bayesian EDA in Polars

Branch: `feat/polars-migration`. State: every table-producing module in
`profiling`, `univariate`, `bivariate` and `statistics` already reads
`dataset.frame` and returns Polars with declared `*_SCHEMA` dicts typed
`dict[str, PolarsDataType]` (`from polars._typing import PolarsDataType`).
Copy those patterns. Read `AGENTS.md` and the "Semánticas" section of
`docs/POLARS_MIGRATION_PLAN.md` first.

You have no terminal. Do not run commands; the reviewer runs lint and tests.
No new modules or abstractions. Statistics stay in NumPy/SciPy as they are.

## Files to change

- `ruddy/analysis/intervals.py`, `ruddy/bayesian/analysis.py`
- `ruddy/bivariate/associations.py`: delete the `removed in task 3F`
  adapter in `_contingency` and narrow its `frame` type back to
  `pl.DataFrame`; drop `import pandas` if unused.
- `ruddy/cli/commands/inference.py`, `ruddy/cli/commands/specialized.py`:
  only pandas idioms on these two results.
- Tests: `tests/analysis/test_intervals.py`, `tests/bayesian/test_bayesian.py`,
  and only assertions on these results in `tests/cli/test_inference_cli.py`,
  `tests/cli/test_specialized_cli.py`, `tests/integration/*`,
  `tests/robustness/*`.

## `analysis/intervals.py`

- `frame = dataset.frame`; numeric arrays via
  `frame.get_column(c).cast(pl.Float64).fill_null(float("nan")).to_numpy()`;
  pairs via `frame.select(x, y).cast(pl.Float64).fill_null(float("nan")).to_numpy()`.
- Two-level groups: labels via `_category_label` on the non-missing rows
  (null or float NaN), `levels = sorted(set(labels))`, per-level arrays with
  NumPy masks, as in `bivariate/comparisons.py`.
- Odds ratios: `_contingency(frame, x, y)` now takes the Polars frame
  directly.
- Today the five tables are built with `pd.DataFrame(rows)` from row dicts
  whose keys **differ between rows** (e.g. skipped rows lack `estimate`).
  Declare one schema per table covering the union of keys, in a stable
  order: `MEANS_SCHEMA`, `CORRELATION_CI_SCHEMA`, `MEAN_DIFFERENCE_SCHEMA`,
  `EFFECT_SIZE_SCHEMA`, `ODDS_RATIO_SCHEMA`. Build with
  `pl.DataFrame(rows, schema=SCHEMA)`; absent keys become null. Replace
  `np.nan` placeholders in the row dicts with `None` (missing is null in
  Polars tables). Types: names/methods/levels/status/reason/`ci_method` →
  String; `n*` → Int64; estimates, standard errors, bounds,
  `confidence_level` → Float64. Also convert non-finite estimates
  (`np.inf`, NaN) coming from the statistics helpers to `None` when storing
  them in the row; `status/reason` already encode the degenerate cases.
- `ConfidenceIntervalResult` fields → `pl.DataFrame`;
  `dataset.n_observations`/`n_columns` → `frame.height`/`frame.width`.

## `bayesian/analysis.py`

- `frame = dataset.frame`; `dataset.columns` → `frame.columns`.
- `values = frame.get_column(column).cast(pl.Float64).fill_null(float("nan")).to_numpy()`
  (the column is numeric by validation, so no `errors="raise"` equivalent is
  needed).
- Group levels: `g = frame.get_column(group)`; levels in first-seen order
  from the non-missing values (`dict.fromkeys`), keeping raw values (they
  are reported as `level_a`/`level_b`); masks `np.array([v == a for v in g.to_list()])`.
- `MEANS_SCHEMA` / `MEAN_DIFFERENCES_SCHEMA` over the union of row keys
  (check `_posterior_summary` for its keys). `level_a`/`level_b` take the
  dtype of the group column: build with `schema_overrides` for all other
  columns as `outliers.py` does for `observation_id`; when every row has
  `None` levels or there are no rows, use the group column dtype for the
  empty/null column (fall back to `pl.String` when there are no groups).
- `BayesianEDAResult` fields → `pl.DataFrame`.

## Tests

Translate as in earlier tasks (`row(0, named=True)`, `.height`, `is None`,
`polars.testing.assert_frame_equal`). Add: a skipped mean row in the
intervals result has `estimate` null and the same `n`; an odds-ratio table
built from a Polars-native dataset has `estimate` as `pl.Float64`.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task pyrefly
uv run task test
```

558 tests plus additions must pass and pyrefly must stay at one error (the
optional `umap` import). If a test outside the listed files fails, name
file, line and idiom. Finish with files touched, tests added, anything not
done.
