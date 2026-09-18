# Task 3E — Grouped responses, general dependence and post-hoc in Polars

Branch: `feat/polars-migration`. State: everything in `ruddy/profiling`,
`ruddy/univariate`, `ruddy/statistics/multiple_testing.py` and the bivariate
core (`correlations`, `comparisons`, `associations`, `contingency`,
`analysis`) already reads `dataset.frame` and returns Polars tables with
declared `*_SCHEMA` dicts. Copy those patterns. Read `AGENTS.md` and the
"Semánticas" section of `docs/POLARS_MIGRATION_PLAN.md` first.

You have no terminal. Do not run commands; the reviewer runs lint and tests.
No new modules or abstractions. Every statistic stays in NumPy/SciPy/sklearn
as it is; only column access and table assembly change.

## Files to change

- `ruddy/bivariate/groups.py`, `dependence.py`, `posthoc.py`
- Remove the `removed in task 3E` adapters in `ruddy/bivariate/comparisons.py`
  and `associations.py` once no caller passes pandas any more (grep
  `ruddy/` for callers first; `analysis/intervals.py` still passes a pandas
  frame to `associations._contingency` — leave **that one adapter** in place
  and relabel it `removed in task 3F`).
- `ruddy/cli/commands/groups.py`, `advanced_groups.py`, `inference.py`: only
  pandas idioms on these results.
- Tests: `tests/bivariate/test_groups.py`, `test_responses.py`,
  `test_dependence.py`, `test_posthoc.py`, `tests/cli/test_groups_cli.py`,
  plus only assertions on these tables in `tests/integration/*`,
  `tests/robustness/*`, `tests/cli/*`.

Out of scope: `analysis/intervals.py`, `bayesian/` (task 3F).

## Common rules

Same as before: `frame = dataset.frame`; `frame.get_column(name)`; finite
numeric values via `cast(pl.Float64).fill_null(float("nan")).to_numpy()` and
`np.isfinite`; category labels via `_category_label` over `.to_list()` of the
non-missing rows (missing = null, or NaN for floats); `<NAME>_SCHEMA` dicts
with `<NAME>_COLUMNS = tuple(SCHEMA)`; `pl.DataFrame(rows, schema=...)` and
`pl.DataFrame(schema=...)` for empty; `dataset.n_observations` →
`dataset.frame.height`.

## `groups.py`

- `_normalized_group(series: pl.Series) -> list[str | None]`: label per row
  (`None` where missing). Levels = `sorted(set(non-None labels))`. Masks are
  NumPy boolean arrays (`np.array([label == level for label in labels])`).
- `summarize_group_coverage`: `coverage_status = {row["group_column"]: row["status"] for row in table.iter_rows(named=True)}`
  replaces the `set_index(...).loc[...]` lookups in the two summary functions.
- Numeric/categorical summaries: slice the response with the NumPy mask
  (`values[mask]` on the float array; `[labels[i] for i in np.flatnonzero(mask)]`
  for categorical labels).
- `_group_counts`: `present = frame.height - n_missing(frame.get_column(group))`
  with the null/NaN rule.
- `_numeric_response_comparisons` / `_categorical_response_comparisons`: the
  inner tables are already Polars. Drop the adapters; if `table.height == 0`
  continue; add the leading columns with
  `table.with_columns(pl.lit(response).alias("response"), pl.lit(dataset.frame.height).cast(pl.Int64).alias("n_dataset"), ...)`
  then `select(_GROUPED_*_COLUMNS)` to fix the order; `family_id` via
  `pl.format("grouped_numeric:{}:{}:{}:{}", pl.lit(group), pl.lit(response), pl.col("scope"), pl.col("test"))`
  (and the categorical analogue); `correction` as `pl.lit(p_adjust.value)`;
  `q_value` as `pl.lit(None, dtype=pl.Float64)`; `family_size` as
  `pl.lit(0, dtype=pl.Int64)`. Combine with `pl.concat(rows, how="vertical")`,
  then `apply_multiple_testing(combined, p_adjust)`. Empty →
  `pl.DataFrame(schema=_GROUPED_*_SCHEMA)`; derive those schemas from the
  core comparison/association schemas plus the added columns.
- `summarize_annotation_coverage` builds from `AlignedAnnotations.summary()`
  dicts: just switch the frame constructor and add its schema.
- `GroupAnalysisResult` fields → `pl.DataFrame`.

## `dependence.py`

- Partial correlations: `matrix = frame.select(columns).cast(pl.Float64).fill_null(float("nan")).to_numpy()`.
- `_encode_categorical(values: list[str]) -> np.ndarray`: codes by
  first-seen order of the string labels
  (`{label: i for i, label in enumerate(dict.fromkeys(labels))}`) — pandas
  `Categorical(...).codes` used **sorted** order; MI is invariant to the
  labelling, so keep whichever is simpler but say which in a comment.
  Callers pass lists of `_category_label` strings for the complete-case rows.
- `_mutual_information(x, y, ...)`: `x`/`y` become NumPy float arrays for
  numeric kinds and label lists for categorical kinds; adapt the four
  branches accordingly. Complete-case selection happens in the caller with a
  NumPy mask (numeric finite, categorical non-missing).
- Tables: `PARTIAL_SCHEMA`, `DEPENDENCE_SCHEMA`; `DependenceResult` → Polars.

## `posthoc.py`

- `_group_values`: `numeric = frame.get_column(response).cast(pl.Float64).fill_null(float("nan")).to_numpy()`;
  `labels = frame.get_column(factor).to_list()`; valid = finite and label
  not None (and not NaN); levels in first-seen order (`dict.fromkeys`) as
  `pd.unique` did. Keep the raw level values as dict keys (they end up in
  `group_a`/`group_b`): decide the `group_a`/`group_b` dtype from the factor
  column — build the table with `schema_overrides` for every column except
  `group_a`/`group_b`, like the outlier flags do for `observation_id`, and use
  the factor dtype for the empty frame.
- The post-processing block that marks small groups (`table.loc[mask, ...]`)
  becomes one `with_columns` with `pl.when(mask_expr).then(...).otherwise(pl.col(...))`
  per affected column, where `mask_expr = pl.col("group_a").is_in(small) | pl.col("group_b").is_in(small)`
  and `small` is the list of too-small levels.
- `PosthocResult.comparisons` → Polars.

## Tests

Translate as in earlier tasks. Add: `analyze_posthoc` on a factor with
integer levels keeps `group_a`/`group_b` as integers (`pl.Int64`), and on a
string factor as `pl.String`; `analyze_grouped_responses` numeric
comparisons have `q_value` null on non-ok rows and `family_size` `Int64`.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task test
```

556 tests plus additions must pass. If a test outside the listed files fails,
name file, line and idiom. Finish with files touched, tests added, adapters
removed/kept, anything not done.
