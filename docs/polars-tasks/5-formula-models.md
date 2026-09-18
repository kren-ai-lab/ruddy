# Task 5 — Formula models: Polars in, explicit pandas boundary to statsmodels/Patsy

Branch: `feat/polars-migration`. State: every other analysis reads
`dataset.frame` and returns Polars tables with declared `*_SCHEMA` dicts;
exclusion tables are built with the helper in
`ruddy/projections/preprocessing.py` (ID dtype preserved). Read `AGENTS.md`
and the plan (`docs/POLARS_MIGRATION_PLAN.md`: "Fronteras con dependencias",
"Detalles verificados que deben conservarse", "Semánticas") first.

You have no terminal. Do not run commands; the reviewer runs lint, pyrefly
and tests. No new modules or abstractions. **statsmodels and Patsy stay.**
This task changes only where data come from and how output tables are
assembled. Nothing in the model formulas, contrasts (`Sum` in factorial and
marginal means, `Treatment` in MANOVA), safe names (`Y`, `F0..`, `X0..`,
`G`), level ordering, or numeric post-processing may change: the existing
tests pin them against statsmodels oracles at `rel=1e-10`.

## Files to change

- `ruddy/factorial/models.py`, `marginal_means.py`, `mixed_effects.py`,
  `diagnostics.py`
- `ruddy/multivariate/manova.py`
- `ruddy/cli/commands/factorial.py`, `advanced_groups.py`, `multivariate.py`
  (MANOVA writer): pandas idioms on these results only.
- Tests: `tests/factorial/*`, `tests/multivariate/test_manova.py`,
  `tests/cli/test_factorial_cli.py`, `test_advanced_groups_cli.py`,
  `test_multivariate_cli.py` (MANOVA part), and only assertions on these
  results in `tests/integration/*`, `tests/robustness/*`.

## The boundary rule

Each module has exactly **one** place where a Polars frame becomes pandas,
right before statsmodels/Patsy, marked with the comment
`# pandas boundary: statsmodels/Patsy consume pandas; see docs/POLARS_MIGRATION_PLAN.md`.
That place is the existing `_safe_*` frame builder:

- `models._safe_model_frame(model_frame: pl.DataFrame, design)`,
  `marginal_means._safe_model(...)`, `mixed_effects._safe_fixed_frame(...)`,
  `manova`'s safe-frame block.
- Inside, do `pandas_frame = model_frame.to_pandas()` first and keep the
  current pandas logic **verbatim** afterwards (`pd.Categorical(...)`,
  `pd.to_numeric(..., errors="raise").astype(float)`, `safe["G"]`, …).
  `pd.Categorical` sorts levels; that ordering is part of the pinned
  output, so do not replace it with a Polars cast.
- Everything statsmodels returns (`params`, `anova_lm` tables, `resid`,
  `mv_test().results`, `design_info`, influence measures) is consumed as
  today and turned into Ruddy's Polars tables in the same module.

## Complete-case selection (Polars)

Replace `frame = dataset.select(selected)` + pandas masks with:

```python
frame = dataset.frame.select(selected)
complete = np.ones(frame.height, dtype=bool)
for name in numeric_columns:
    values = frame.get_column(name).cast(pl.Float64).fill_null(float("nan")).to_numpy()
    complete &= np.isfinite(values)
for name in factor_columns:  # and the mixed-effects group column
    complete &= frame.get_column(name).is_not_null().to_numpy()
model_frame = frame.filter(pl.Series(complete))
```

A float factor column with NaN counts as missing too (`& ~is_nan()` for
float dtypes). Exclusions via the shared helper with the module's `stage`
and `reason` strings. IDs from `dataset.observation_id_tuple`;
`ids.take(idx)` → `tuple(ids[i] for i in idx)`.

## Output tables

- Every `pd.DataFrame(rows, columns=...)` / `pd.DataFrame(columns=...)`
  becomes `pl.DataFrame(rows, schema=<SCHEMA>)` / `pl.DataFrame(schema=<SCHEMA>)`
  with a schema dict next to each `*_COLUMNS` tuple (`tuple(SCHEMA)`).
  Names/terms/levels/status/reason/stage → String; counts, df (integers) →
  Int64; SS, MS, F, p, q, estimates, SE, CIs, leverage, Cook's D, ICC,
  variances, log-likelihood → Float64; booleans → Boolean. Where a value can
  be missing, store `None` (not `np.nan`).
- `diagnostics.py`: the cell table has factor-level columns whose dtype is
  the factor's dtype: build with `schema_overrides` for the fixed columns
  and let the level columns infer (empty variant: use the factor dtypes from
  `model_frame.schema`). `pd.unique(frame[factor])` →
  `list(dict.fromkeys(frame.get_column(factor).to_list()))`.
  `observation_diagnostics` `observation_id` keeps the ID dtype.
- Marginal means `means`/`contrasts`: level values keep the factor dtype
  (same `schema_overrides` technique).
- Mixed effects: `group_counts = model_frame.get_column(group).value_counts()`
  → read counts via `iter_rows(named=True)` or `.to_dict()`; keep the
  reported numbers identical. Random-effects / variance-component tables →
  schemas; safe names `X0` etc. stay as reported today.
- MANOVA: the statistics table (`mv_test().results[...]["stat"]`, a pandas
  frame per term) is flattened into the existing row dicts; keep every
  value and the `Intercept` row.
- Result dataclasses: all `pd.DataFrame` fields → `pl.DataFrame`.

## `models.py` specifics

- `_design_term_table`, `effects`, `coefficients`: schemas; coefficient
  names keep Patsy's `[S.level]` suffixes.
- `response_values = model_frame.get_column(design.response).cast(pl.Float64).to_numpy()`.
- The influence/observation diagnostics receive `observation_ids` as a
  tuple and `source_row_indices` as today.

## Tests

Translate as in earlier tasks. Oracle comparisons against statsmodels stay
exactly as they are (they import statsmodels directly and compare numbers).
Add: `analyze_factorial` on a Polars-native dataset with an integer-coded
factor declared as factor produces the same `effects` table as the same
data given as pandas (`polars.testing.assert_frame_equal`); exclusions
`observation_id` keeps the dataset's ID dtype.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task pyrefly
uv run task test
grep -rn "import pandas" ruddy/factorial ruddy/multivariate/manova.py
```

The grep must show only the modules that own a boundary (the four factorial
modules and manova). All tests plus additions must pass at their current
tolerances; pyrefly stays at one error (optional `umap` import). If an
oracle test fails, do not loosen it and do not touch the model call: report
term, observed and expected values. If a test outside the listed files
fails, name file, line and idiom. Finish with files touched, tests added,
anything not done.
