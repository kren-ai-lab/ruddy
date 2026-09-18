# Task 4B — Representation comparison and compositional analysis in Polars

Branch: `feat/polars-migration`. State: everything except `representation/`,
`compositional/`, `multivariate/manova.py` and `factorial/` already returns
Polars. Copy the patterns from `ruddy/projections/preprocessing.py`
(`_exclusions_table`), `ruddy/projections/pca.py` (scores/loadings tables)
and `ruddy/multivariate/covariance.py` (`square_table`). Read `AGENTS.md`
and the plan (`docs/POLARS_MIGRATION_PLAN.md`, "Contrato objetivo" and
"Semánticas") first.

You have no terminal. Do not run commands; the reviewer runs lint, pyrefly
and tests. No new modules or abstractions. Every numerical computation stays
in NumPy/SciPy/sklearn.

## Files to change

- `ruddy/representation/analysis.py`, `ruddy/compositional/analysis.py`
- `ruddy/cli/commands/specialized.py`: representation and compositional
  writers only (drop `index=True`, `.empty`, the pandas `transformed`
  assembly).
- Tests: `tests/representation/test_representation.py`,
  `tests/compositional/test_compositional.py`,
  `tests/robustness/test_representation_torture.py`,
  `tests/robustness/test_compositional_and_anomaly_torture.py`,
  `tests/cli/test_specialized_cli.py` (only these two writers), and only
  assertions on these results in `tests/integration/*`.

## `representation/analysis.py`

- `AlignedRepresentationPair.observation_ids: tuple[ObservationID, ...]`;
  `align_feature_matrices` works on `x.observation_id_tuple` /
  `y.observation_id_tuple` (common IDs in x order as today), exclusions via
  the shared `_exclusions_table`-style helper (import it from
  `projections.preprocessing` if it is reusable as is; otherwise build the
  table with `schema_overrides` and the ID dtype like there).
- CCA tables: weights/loadings as `feature` String + component columns
  (`pl.DataFrame({"feature": names, **{c: col for ...}})`); scores as
  `observation_id` (ID dtype) + components. Correlations table from its row
  dicts with a `CCA_CORRELATION_SCHEMA`. The empty `CCAResult` variants use
  those schemas with zero rows instead of bare `pd.DataFrame()`.
- CKA, Procrustes, distance similarity, Mantel tables: schema dicts over
  their row keys; `None` instead of `np.nan` for missing.
- `RepresentationComparisonResult` fields → `pl.DataFrame`.

## `compositional/analysis.py`

- `variation_matrix(array, feature_names) -> pl.DataFrame`:
  `square_table(out, feature_names, label_column="feature")`.
- `aitchison_distance_matrix(array, observation_ids: Sequence) -> pl.DataFrame`:
  `square_table(dist, observation_ids, label_column="observation_id")`.
- `multiplicative_zero_replacement` returns a Polars table with a
  `ZERO_REPLACEMENT_SCHEMA`; the empty variant in `analyze_composition` too.
- `CompositionalResult` fields → `pl.DataFrame`; IDs via
  `features.observation_id_tuple`.

## CLI (`specialized.py`)

- `transformed = pl.DataFrame(result.transformed.to_array(), schema=list(result.transformed.feature_names))`
  then `.with_columns(pl.Series("observation_id", result.transformed.observation_id_tuple))`
  and `.select("observation_id", *feature_names)`.
- `write_table(result.variation_matrix, ...)` and
  `write_table(result.aitchison_distances, ...)` without `index=True`.
- `if result.exclusions.height:`.

## Tests

Translate as before. Add: `aitchison_distances` on a `FeatureMatrix` with
string IDs has `observation_id` dtype `pl.String` and its diagonal is zero
(`table.row(i)[i + 1] == 0.0`); `variation_matrix` columns are
`["feature", *feature_names]`.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task pyrefly
uv run task test
```

562 tests plus additions must pass; pyrefly stays at one error (optional
`umap` import). If a test outside the listed files fails, name file, line
and idiom. Finish with files touched, tests added, anything not done.
