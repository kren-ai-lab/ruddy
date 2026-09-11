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

`TabularDataset` copies the input `DataFrame` and exposes defensive copies. Constructing a dataset does not coerce values, impute data or remove observations.

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

Observation identifiers must be non-missing and unique. If `id_column` is not supplied, the DataFrame index is used as observation identity.

Ruddy also requires unique string column names so schemas remain stable across serialization and downstream tooling.

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
- numeric pandas DataFrames;
- SciPy sparse matrices.

A DataFrame supplied to `FeatureMatrix` must contain numeric columns only.

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

Annotation coverage is represented as `complete`, `partial` or `absent` and can be summarized as a table for reporting.

## Non-finite values

Ruddy distinguishes:

- missing values such as `NaN`;
- present but non-finite numeric values such as `+inf` and `-inf`.

Numeric analysis generally uses finite observations only, while missing/non-finite counts remain separate in outputs.

For `FeatureMatrix` analyses that require complete finite rows, excluded rows are recorded with:

- observation ID;
- source row index;
- analysis stage;
- exclusion reason.

## Immutability and non-destructive behavior

The scientific input is immutable by contract:

- constructors copy caller-owned data;
- public data accessors return copies;
- analyses return new result objects;
- outlier/anomaly analyses never mutate the input;
- transformed compositional/PCA spaces are returned as new feature matrices.

This makes it possible to trace every derived object back to an unchanged input state.
