# Task 6B — marimo examples on the Polars API

Branch: `feat/polars-migration`. The library now stores Polars in
`TabularDataset.frame`, returns Polars tables from every analysis, and its
public surface changed (see `docs/polars-tasks/6A-remove-pandas-adapters.md`,
section "Target public contract"). The examples under `examples/` still use
the pandas-era API. Read `AGENTS.md` (section "Core vs. Visualization") and
`examples/README.md` first.

You have no terminal. Do not run commands; the reviewer runs
`bash examples/run_ci_examples.sh`. Do not touch anything under `ruddy/` or
`tests/`.

## Files to change

- `examples/_helpers.py`, `examples/01_*.py` … `examples/13_*.py`,
  `examples/data/generate_demo_data.py` (only if it uses a changed API),
  `examples/README.md` (only the lines describing inputs/outputs).

## Rules

- Load demo data with Polars: `pl.read_csv(...)`; for `tabular_demo.csv`
  parse `recorded_at` as datetime (`try_parse_dates=True` or an explicit
  `str.to_datetime()`), then build `TabularDataset(frame, id_column=...)`.
  `FeatureMatrix(frame.select(numeric_columns), observation_ids=frame["id"].to_list(), ...)`.
- Result tables are Polars. Replace pandas idioms on them:
  `.iloc[0].x` → `.row(0, named=True)["x"]` or `.item(0, "x")`;
  `.query(...)` → `.filter(pl.col(...) == ...)`; `.set_index(...)[cols]` →
  `.select(...)`; `.iterrows()` → `.iter_rows(named=True)`; `.empty` →
  `.height == 0`; `.pivot(...)` → `.pivot(on=..., index=..., values=...)`;
  `.groupby(...)` → `.group_by(...)`.
- Square matrices from Ruddy (covariance, correlations, distances,
  variation, Aitchison) are Polars tables whose first column is the row
  label (`feature` or `observation_id`) and whose other columns are the
  matrix. When a helper needs a NumPy matrix plus labels, take
  `labels = table.get_column(table.columns[0]).to_list()` and
  `values = table.select(table.columns[1:]).to_numpy()`.
- Plotting helpers in `_helpers.py` (matplotlib/plotly) may keep working on
  pandas internally: convert at the call site with `.to_pandas()` **only**
  where a helper genuinely needs pandas (e.g. seaborn-style long frames);
  prefer changing the helper to accept a Polars frame or NumPy arrays when
  that is a one-line change. Never recompute a statistic in an example that
  Ruddy already reports (that is a project rule).
- Small ad-hoc tables built in the examples for display (`pd.DataFrame(rows)`)
  become `pl.DataFrame(rows)`; `display(...)` works on Polars frames in
  marimo.
- Keep every example's narrative and figures; this is a mechanical port.

## Acceptance (reviewer)

```bash
bash examples/run_ci_examples.sh
```

Every example must run to completion headless (the script sets
`MPLBACKEND=Agg`). Finish with files touched, every place where you kept a
`.to_pandas()` and why, anything not done.
