# Task 6C — Documentation for the Polars API

Branch: `feat/polars-migration`. The library now stores Polars in
`TabularDataset.frame` and returns Polars tables from every analysis; pandas
remains accepted as *input* to `TabularDataset`/`FeatureMatrix` and is used
internally only at four explicit statsmodels/Patsy boundaries. The public
surface changed as described in
`docs/polars-tasks/6A-remove-pandas-adapters.md` ("Target public contract")
and the labeled-matrix contract in `docs/POLARS_MIGRATION_PLAN.md`
("Contrato objetivo"). Read `AGENTS.md` first and verify every claim you
write against the code in `ruddy/` (signatures, attribute names, dtypes).

You have no terminal. Do not run commands. Do not touch `ruddy/`, `tests/`
or `examples/`.

## Files to change

- `README.md`, `DEVELOPMENT.md` (dependencies: polars, pyarrow; pandas only
  as an accepted input / statsmodels transitive), `AGENTS.md` (the layout and
  invariants sections if they mention pandas or `to_frame`).
- `docs/PUBLIC_API.md` (the bulk: every `pd.DataFrame` → `pl.DataFrame`;
  remove `to_frame()`, `select()`, `n_observations`, `n_columns`, `columns`,
  `observation_id_tuple`; add `frame`, `provenance`, `observation_ids` as a
  tuple, `observation_ids=` constructor argument; `align_annotations`
  requires `id_column`; `read_table`/`write_table` on Polars).
- `docs/DATA_CONTRACTS.md`, `docs/OUTPUT_SCHEMAS.md` (tables now carry
  declared schemas; missing values are null, never NaN; labeled matrices
  have the row label in the first column; `observation_id` columns keep the
  dataset's ID dtype), `docs/RESULTS_AND_PROVENANCE.md`,
  `docs/STATISTICAL_POLICIES.md` (missingness policy: null or float NaN
  count as missing; skewness/kurtosis via SciPy bias-corrected estimators;
  Spearman via average ranks), `docs/METHOD_INVENTORY.md`,
  `docs/CLI_REFERENCE.md` (matrix CSVs now have a `feature`/`observation_id`
  first column instead of a pandas index), `docs/TESTING_AND_REPRODUCIBILITY.md`,
  `docs/VISUALIZATION_EXAMPLES.md`, `examples/README.md` if it documents
  API types.
- Add a short "Breaking changes" section to `README.md` or a new
  `docs/CHANGELOG_POLARS.md` (pick the one that fits the repo's style)
  listing: Polars result tables; `TabularDataset` surface; pandas index is
  never identity; complex dtypes rejected; null instead of NaN; labeled
  matrix layout; `ColumnSpec.dtype` strings are Polars dtype names.

Leave `docs/POLARS_MIGRATION_PLAN.md`, `docs/POLARS_MIGRATION_REVIEW_FABLE.md`
and `docs/POLARS_REPLACEMENT_ESTIMATES.md` alone except for updating the
plan's status line to "ejecutado; pendiente revisión final".

## Acceptance (reviewer)

`grep -rn "pd\.DataFrame\|to_frame()\|index=True" README.md DEVELOPMENT.md docs/*.md examples/README.md`
must return nothing outside the three migration documents. Finish with files
touched and anything you were unsure about (with the code location you
checked).
