# Task 4A — Projections, anomaly and multivariate matrices in Polars

Branch: `feat/polars-migration`. State: every tabular analysis in
`profiling`, `univariate`, `bivariate`, `statistics`, `analysis/intervals`
and `bayesian` reads `dataset.frame` and returns Polars tables with declared
`*_SCHEMA` dicts (`dict[str, PolarsDataType]`,
`from polars._typing import PolarsDataType`). `FeatureMatrix` stores IDs as
a tuple (`features.observation_id_tuple`; the `observation_ids` property
still returns a temporary `pd.Index`). Read `AGENTS.md` and the plan
(`docs/POLARS_MIGRATION_PLAN.md`, sections "Contrato objetivo" and
"Semánticas") first.

You have no terminal. Do not run commands; the reviewer runs lint, pyrefly
and tests. No new modules or abstractions. Every numerical computation stays
in NumPy/SciPy/sklearn; only ID handling and table assembly change.

## Labeled-matrix contract (decided 2026-09-18)

A square matrix with labels on both axes is a Polars table whose **first
column holds the row label with its original dtype** and whose remaining
columns are the matrix values, named `str(label)`:

- feature × feature (covariance, Pearson, Spearman, pairwise counts, VIF
  design, compositional variation): first column `feature` (`pl.String`).
- observation × observation (distance matrices, Aitchison): first column
  `observation_id`, dtype of the IDs.

Add one helper in `ruddy/multivariate/covariance.py` (it already has
`_square_frame`) and import it where needed:

```python
def square_table(values: np.ndarray, labels: Sequence[Any], *, label_column: str) -> pl.DataFrame:
    matrix = np.asarray(values, dtype=float)
    columns = {label_column: pl.Series(label_column, list(labels))}
    for j, label in enumerate(labels):
        columns[str(label)] = pl.Series(str(label), matrix[:, j], dtype=pl.Float64)
    return pl.DataFrame(columns)
```

Integer pairwise counts use `dtype=pl.Int64` (make the dtype a parameter or
cast after). Name collisions after `str()` cannot happen for feature names
(they are unique strings) and are prevented for IDs by ID uniqueness.

## Files to change

- `ruddy/projections/preprocessing.py`, `pca.py`, `manifold.py`
- `ruddy/anomaly/analysis.py`
- `ruddy/multivariate/covariance.py`, `collinearity.py`, `distances.py`,
  `permutation.py`
- `ruddy/cli/commands/projections.py`, `multivariate.py`,
  `advanced_groups.py` (PERMANOVA writer), `specialized.py` (anomaly writer
  if there): drop `index=True`/`index_label=...` on matrix writes, replace
  `.empty`/`shape[0]`/`.iloc` idioms.
- Tests: `tests/projections/*`, `tests/anomaly/*`, `tests/multivariate/test_covariance*.py`,
  `test_collinearity*.py`, `test_distances*.py`, `test_permutation*.py`
  (whatever exists for these modules), `tests/cli/test_projections_cli.py`,
  `test_multivariate_cli.py`, `test_advanced_groups_cli.py`, and only
  assertions on these results in `tests/integration/*`, `tests/robustness/*`.

Out of scope: `representation/`, `compositional/`, `multivariate/manova.py`,
`factorial/` (tasks 4B and 5).

## `projections/preprocessing.py`

- `PreparedFeatures.observation_ids: tuple[ObservationID, ...]` (use
  `features.observation_id_tuple`; selection by positions with
  `tuple(ids[i] for i in source_rows)`).
- `exclusions` → Polars with `EXCLUSIONS_SCHEMA_BASE`
  (`source_row_index` Int64, `stage` String, `reason` String) and
  `observation_id` inferred from the IDs (`schema_overrides`, and for zero
  rows the ID dtype via `pl.Series(ids).dtype`, exactly as
  `univariate/outliers.py` does for its flags). Put that small helper
  (`_exclusions_table(ids, excluded_rows)`) in `preprocessing.py`; the
  other modules that build exclusion tables reuse it.

## `projections/pca.py`, `manifold.py`, `anomaly/analysis.py`

- Scores tables: `pl.DataFrame` with `observation_id` first (ID dtype),
  `source_row_index` Int64 where present, then component/score columns
  Float64. Loadings: `feature` String + components. Variance table: schema
  from its current columns. Empty results: same schemas with zero rows.
- `PCAResult.to_feature_matrix`: `observation_ids=self.scores.get_column("observation_id").to_list()`.
- Anomaly `scores`/`flags`: same pattern as outlier flags.
- Every `pd.Index.take(...)`/`.to_list()` on IDs becomes tuple indexing.

## `multivariate/covariance.py`

- Square tables via `square_table(..., label_column="feature")`;
  `pairwise_counts` Int64.
- **Spearman without pandas**: the matrix is complete-case finite, so
  `spearman = np.corrcoef(stats.rankdata(matrix, axis=0), rowvar=False)`
  (average ranks, like pandas) inside the same `np.errstate` guard as
  Pearson; constant columns yield NaN as before. `np.atleast_2d`. Keep the
  `include_spearman=False` NaN fill.
- `feature_diagnostics`, `condition_spectrum`, `exclusions` → Polars with
  schemas. `n_nonconstant = feature_diagnostics.filter(~pl.col("is_constant")).height`.

## `multivariate/collinearity.py`, `distances.py`, `permutation.py`

- Same rules. `distances.py`: the `pd.concat(distance_frames)` becomes
  `pl.concat(frames, how="vertical")` (or a single `pl.DataFrame(rows, schema=...)`
  if rows are dicts); distance matrices via
  `square_table(..., label_column="observation_id")`.
- `permutation.py`: `_aligned_factor` already receives a Polars series from
  `align_annotations`; return `aligned.get_column(factor)` (a `pl.Series`)
  and drop the `removed in phase 5` adapter there; labels for grouping via
  `.to_list()`. PERMANOVA/PERMDISP result tables → Polars with schemas.

## CLI

`write_table(matrix, path)` with no `index` arguments: the label column is
already the first column. Row counts via `.height`.

## Tests

Translate as in earlier tasks. Add: `square_table` round-trip — a 3×3
covariance on features `("a", "b", "c")` has columns
`["feature", "a", "b", "c"]`, `feature` dtype `pl.String`, and
`table.filter(pl.col("feature") == "b").row(0)[2]` equals the diagonal
value; a distance matrix on integer IDs has `observation_id` dtype
`pl.Int64` and column names `["observation_id", "0", "1", ...]`. Also:
Spearman via ranks equals `scipy.stats.spearmanr(matrix).statistic` on a
random 20×4 matrix within 1e-12.

## Acceptance (reviewer)

```bash
uv run task lint
uv run task pyrefly
uv run task test
```

560 tests plus additions must pass; pyrefly stays at one error (optional
`umap` import). If a parity test fails on Spearman, do not loosen
tolerances: report the values. If a test outside the listed files fails,
name file, line and idiom. Finish with files touched, tests added, anything
not done.
