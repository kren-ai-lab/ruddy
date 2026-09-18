# Data contracts and alignment

Ruddy separates the statistical meaning of a column from its observed storage type and treats observation identity as a first-class scientific contract.

## `TabularDataset`

```python
from ruddy import TabularDataset

dataset = TabularDataset(
    frame,
    id_column="id",
    role_overrides={
        "activity": "response",
        "family": "factor",
        "length": "covariate",
        "sequence": "excluded",
    },
)
```

`TabularDataset` stores tabular data internally as a Polars DataFrame (`pl.DataFrame`) exposed via the `.frame` property. Inputs may be either Polars or pandas DataFrames. When constructed from pandas, the frame is deep-copied and converted via Arrow without index. Constructing a dataset does not coerce values, impute data or silently remove observations. Temporary adapter methods (frame export, `select()`, `n_observations`, `n_columns`, `columns`, `__len__`, `observation_id_tuple`) have been removed; callers access `dataset.frame.height`, `dataset.frame.width`, `dataset.frame.columns`, and `dataset.frame.select(...)`.

### Column roles

| Role | Meaning |
| --- | --- |
| `identifier` | Observation identity; excluded from statistical analysis |
| `variable` | General analyzable variable |
| `response` | Variable explicitly treated as a response in response-centric analyses |
| `factor` | Grouping/factor variable; treated categorically even when numerically encoded |
| `covariate` | Numeric adjustment/model variable |
| `annotation` | Contextual aligned metadata |
| `excluded` | Retained in the source table but excluded from analyses |

A crucial design decision is that a numerically encoded column can be a factor:

```python
TabularDataset(
    frame,
    id_column="id",
    role_overrides={"batch": "factor"},
)
```

`batch = 0, 1, 2` is then treated as categorical grouping information rather than a continuous trend.

### Observed data kinds

| Kind | Meaning |
| --- | --- |
| `numeric` | Numeric dtype suitable for numeric analysis |
| `categorical` | Nominal/categorical values |
| `boolean` | Boolean data kept distinct from generic categorical data |
| `datetime` | Datetime data for descriptive temporal range summaries |
| `unknown` | Unsupported/unknown observed kind |

Roles and kinds are distinct. A factor may have numeric kind, and a response may be numeric or categorical.

### Identity requirements

Observation identifiers must be non-missing and unique. Identity can be established in three ways:
1. `id_column`: Specified column from the table.
2. `observation_ids`: Explicit sequence of identifiers matching `frame.height`.
3. Generated integer IDs: When neither is provided, default IDs `0..n-1` are generated (`range(n)`).

**A pandas index is never treated as identity.** If a pandas DataFrame with a non-default index is passed without `id_column` or `observation_ids`, Ruddy raises a `ValueError` requiring the caller to explicitly provide `observation_ids=frame.index` or call `frame.reset_index()`.

Observation IDs are exposed as an immutable tuple via `dataset.observation_ids`.

Ruddy requires unique string column names so schemas remain stable across serialization and downstream tooling. Columns with complex number dtypes (e.g. `complex64`, `complex128`) are rejected with a `TypeError`.

## `FeatureMatrix`

`FeatureMatrix` represents an observations × features numerical space.

```python
from ruddy import FeatureMatrix

features = FeatureMatrix(
    matrix,
    observation_ids=ids,
    feature_names=feature_names,
)
```

Accepted inputs include:

- NumPy 2D arrays;
- numeric Polars DataFrames;
- numeric pandas DataFrames;
- SciPy sparse matrices.

A DataFrame supplied to `FeatureMatrix` must contain numeric columns only. Observation IDs are exposed as an immutable tuple via `features.observation_ids` (`observation_id_tuple` has been removed). Optional metadata is stored as a Polars DataFrame (`features.metadata`).

### Feature names

If names are not supplied, Ruddy generates stable names:

```text
feature_0
feature_1
...
```

Names must be unique.

### Dense and sparse behavior

Sparse support is method-specific. Ruddy preserves sparse input when a method can safely operate on it and rejects operations that would require hidden densification.

Examples:

- standard/robust preprocessing can support sparse matrices without centering where appropriate;
- PCA as currently implemented requires dense input and will not silently densify a large sparse representation;
- pairwise representation comparison currently requires dense input.

The user must make any memory-expensive dense conversion explicitly.

## Feature metadata

A `FeatureMatrix` can receive metadata aligned by observation ID:

```python
features = FeatureMatrix(
    matrix,
    observation_ids=ids,
    metadata=metadata,
    metadata_id_column="id",
    metadata_alignment="strict",
)
```

The resulting alignment report remains accessible.

## Strict versus partial alignment

Ruddy supports two alignment policies.

### `strict`

Requires exact one-to-one ID coverage. Missing or unmatched observation IDs produce an alignment error.

Use strict alignment when the two objects are expected to describe exactly the same observation set.

### `partial`

Uses the overlapping observations while retaining an explicit report of:

- base count;
- incoming/annotation count;
- covered count;
- coverage fraction;
- missing IDs;
- unmatched IDs.

Partial alignment is never a silent inner join.

## External annotations

External annotation tables can be aligned before group-wise analysis:

```python
from ruddy import align_annotation_source, attach_annotations

aligned = align_annotation_source(
    dataset,
    metadata,
    source_name="experimental_metadata",
    id_column="id",
    mode="partial",
    role_overrides={"cohort": "factor"},
)

extended = attach_annotations(dataset, aligned)
```

`align_annotations` and `align_annotation_source` require an explicit `id_column`. A pandas DataFrame without `id_column` raises `ValueError("Annotations require id_column; a pandas index is not an identity. Pass frame.reset_index() with the ID as a column.")`. Aligned annotations store data as a Polars DataFrame (`aligned.frame`).

Annotation coverage is represented as `complete`, `partial` or `absent` and can be summarized as a table for reporting.

## Non-finite values and missingness

Ruddy distinguishes:

- missing values: represented as Polars `null` (or float `NaN` in source floating-point series);
- present but non-finite numeric values such as `+inf` and `-inf`.

In numeric descriptive analyses, both Polars `null` and float `NaN` count as missing (`n_missing`), while `+inf`/`-inf` are recorded under `n_non_finite`. Result tables use Polars `null` exclusively for missing values and never output `NaN`.

Numeric analysis generally uses finite observations only, while missing/non-finite counts remain separate in outputs.

For `FeatureMatrix` analyses that require complete finite rows, excluded rows are recorded in a Polars DataFrame with:

- `observation_id` (preserving the dataset's observation ID dtype);
- `source_row` index;
- `stage`;
- `reason`.

## Labeled and square matrix contracts

Square matrices (covariance, Pearson/Spearman correlation matrices, pairwise counts, variation matrix, Aitchison distances) and projection tables (scores, loadings, coordinates) are represented as Polars DataFrames:
- The first column is named `feature` (for feature-by-feature matrices) or `observation_id` (for observation-indexed matrices).
- The first column retains the dataset's or matrix's label/ID dtype.
- The remaining columns contain matrix values and are named `str(label)`.
- Exporting to CSV via `write_table` or the CLI outputs this table directly without writing an unnamed index column.

## Immutability and non-destructive behavior

The scientific input is immutable by contract:

- constructors copy caller-owned data (Polars frames are immutable; pandas inputs are deep-copied on ingestion);
- public data accessors return Polars frames;
- analyses return new result objects containing Polars result tables;
- outlier/anomaly analyses never mutate the input;
- transformed compositional/PCA spaces are returned as new feature matrices.

This makes it possible to trace every derived object back to an unchanged input state.
