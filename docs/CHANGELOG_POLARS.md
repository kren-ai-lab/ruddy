# Migration to Polars — Breaking Changes

This document records the breaking changes introduced by migrating Ruddy's internal storage and public result contracts from pandas to [Polars](https://pola.rs/).

## Summary of Changes

### 1. Polars Result Tables
Every analysis and diagnostic method in Ruddy now returns tabular results as Polars DataFrames (`polars.DataFrame` / `pl.DataFrame`) rather than pandas DataFrames.
- Tables carry explicitly declared column schemas and dtypes, even when empty or containing all-null columns.
- Downstream code accessing table rows and columns should use Polars expressions (e.g., `df.select(...)`, `df.filter(...)`) or convert defensively via `df.to_numpy()` or `df.to_pandas()`.

### 2. `TabularDataset` Surface
The public contract of `TabularDataset` has been simplified and aligned with Polars:
- **`dataset.frame`**: The underlying Polars DataFrame is exposed directly via the read-only `.frame` property.
- **`dataset.observation_ids`**: Now returns an immutable `tuple[ObservationID, ...]`.
- **`observation_ids=` parameter**: The constructor now accepts an explicit `observation_ids: Sequence[ObservationID] | None = None` argument alongside `id_column`.
- **Removed members**: The temporary pandas adapter methods and properties have been removed:
  - Legacy frame export method — use `dataset.frame` (or `dataset.frame.to_pandas()` if pandas is required);
  - `select(columns)` — use `dataset.frame.select(columns)`;
  - `n_observations` — use `dataset.frame.height`;
  - `n_columns` — use `dataset.frame.width`;
  - `columns` — use `dataset.frame.columns`;
  - `__len__` — use `dataset.frame.height`;
  - `observation_id_tuple` — use `dataset.observation_ids`.

### 3. pandas Index Is Never Identity
Observation identity is an explicit scientific contract in Ruddy.
- A pandas DataFrame passed to `TabularDataset` must either have a default 0-indexed `RangeIndex`, or the caller must explicitly specify `id_column="col"` or `observation_ids=frame.index`. A non-default pandas index without explicit ID raises a `ValueError`.
- In `align_annotations` and `align_annotation_source`, annotations must have an explicit `id_column`. A pandas DataFrame without `id_column` raises `ValueError("Annotations require id_column; a pandas index is not an identity. Pass frame.reset_index() with the ID as a column.")`.

### 4. Complex Data Types Rejected
Polars does not support complex number dtypes (`complex64`, `complex128`). Constructing a `TabularDataset` from a pandas DataFrame containing complex dtypes raises a `TypeError`. Complex columns must be dropped or converted explicitly before dataset creation.

### 5. Null Instead of NaN
Missing tabular values are stored and returned as Polars `null`, never float `NaN`.
- In numeric univariate statistics, both Polars `null` and float `NaN` are accounted for under `n_missing`. Non-finite values (`+inf`, `-inf`) are counted separately under `n_non_finite`.
- Output tables use `null` for missing statistics or skipped metrics.

### 6. Labeled and Square Matrix Layout
Square and labeled matrices (such as covariance, Pearson/Spearman correlation matrices, pairwise counts, variation matrices, Aitchison distances, and projection scores/loadings) are returned as Polars DataFrames:
- The first column is named `feature` (for feature-by-feature matrices) or `observation_id` (for observation-by-observation matrices or scores).
- The first column preserves the original label or observation ID dtype.
- Subsequent columns contain the matrix values and are named `str(label)`.
- CSV export via `ruddy.core.io.write_table` or the CLI writes the table directly without an unlabelled index column.

### 7. `ColumnSpec.dtype` Strings Are Polars Dtype Names
`ColumnSpec.dtype` in `dataset.schema` now records Polars data type names (such as `"Int64"`, `"Float64"`, `"String"`, `"Boolean"`, `"Date"`, `"Datetime"`) rather than pandas or NumPy dtype strings (such as `"int64"`, `"float64"`, `"object"`).

### 8. Dependency Changes
- `polars` (>=1.44) and `pyarrow` (>=25) are core dependencies.
- `pandas` remains a direct dependency for accepted inputs to `TabularDataset` and `FeatureMatrix` and the explicit statsmodels/Patsy model boundaries.
