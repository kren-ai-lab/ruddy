# Data contracts and alignment

Use `TabularDataset` for columns with statistical roles, and `FeatureMatrix`
for numerical representations. Supply observation IDs whenever objects will
be combined; matching row positions do not establish identity.

## Tabular datasets

```python
import polars as pl
from ruddy import TabularDataset

frame = pl.DataFrame({
    "id": ["s1", "s2", "s3", "s4"],
    "activity": [1.2, 2.4, 1.8, 3.1],
    "batch": [0, 0, 1, 1],
})
dataset = TabularDataset(
    frame,
    id_column="id",
    role_overrides={"activity": "response", "batch": "factor"},
)
print(dataset.frame.select("activity"))
print(dataset.observation_ids)
print(dataset.schema)
```

Inputs can be Polars or pandas DataFrames. `dataset.frame` is a Polars
DataFrame; use its `.height`, `.width`, `.columns` and `.select(...)` for table
operations. `dataset.schema` records each column's role, kind and Polars dtype
name, such as `Float64` or `String`.

Column names must be unique strings. Unsupported data types, including complex
numbers and pandas categoricals with non-string categories, require an explicit
conversion before construction. Numeric-looking strings are not converted to
numbers; missing values are not imputed and rows are not dropped.

### Roles and kinds

A role states how a column is used. A kind describes its observed data type.
For example, the numeric `batch` column above is treated categorically because
its role is `factor`.

| Role | Use |
| --- | --- |
| `identifier` | Observation identity; excluded from analysis |
| `variable` | General analyzable variable |
| `response` | Outcome selected for response-centric analyses |
| `factor` | Categorical grouping or model factor |
| `covariate` | Numeric adjustment/model variable |
| `annotation` | Contextual aligned metadata |
| `excluded` | Retained in the table, excluded from analysis |

Kinds are `numeric`, `categorical`, `boolean`, `datetime` and `unknown`.
Use `kind_overrides` when an explicit statistical kind is needed; an override
does not convert the underlying values. All-null columns need an explicit
input dtype to establish that they are numeric.

Inspect selections with `dataset.role_of(name)`, `dataset.kind_of(name)`,
`dataset.columns_with_role(...)` and `dataset.columns_with_kind(...)`.

### Observation identity

IDs must be non-missing and unique. For `TabularDataset`, choose either:

- `id_column="id"` to use a table column;
- `observation_ids=ids` to provide a sequence matching the number of rows.

Passing both raises an error. Without either, Ruddy generates integer IDs
`0..n-1`. IDs are exposed as an immutable tuple in `dataset.observation_ids`.
Generated IDs only identify rows within that dataset; supply shared IDs before
combining independently loaded objects.

A pandas DataFrame with a non-default index requires an explicit identity
choice for `TabularDataset`: pass `observation_ids=frame.index`, or use
`frame.reset_index()` and specify the resulting `id_column`.

## Feature matrices

`FeatureMatrix` accepts a two-dimensional NumPy array, a numeric Polars or
pandas DataFrame, or a SciPy sparse matrix. DataFrame inputs must contain only
numeric features; provide identifiers separately.

```python
from ruddy import FeatureMatrix

features = FeatureMatrix(
    dataset.frame.select("activity"),
    observation_ids=dataset.observation_ids,
)
print(features.shape)
print(features.feature_names)
```

Feature names come from DataFrame columns or an explicit `feature_names`
sequence. Otherwise they are generated as `feature_0`, `feature_1`, and so on.
Names must be unique. Pass `observation_ids` explicitly for consistent identity
across all input formats. Access the validated tuple as `features.observation_ids`.

Sparse support depends on the method. Preprocessing preserves sparse data where
supported; PCA and representation comparison require dense inputs. Operations
that would need implicit densification are rejected. Any dense conversion must
be an explicit caller decision.

## Align annotations and metadata

Annotation tables require an explicit ID column. They are aligned by ID,
regardless of input row order.

```python
from ruddy import align_annotation_source, attach_annotations

metadata = pl.DataFrame({"id": ["s4", "s1"], "source": ["B", "A"]})
aligned = align_annotation_source(
    dataset,
    metadata,
    source_name="sample_metadata",
    id_column="id",
    mode="partial",
    role_overrides={"source": "factor"},
)
extended = attach_annotations(dataset, (aligned,))
print(aligned.report)
```

| Mode | Behavior |
| --- | --- |
| `strict` | Requires exact one-to-one ID coverage; mismatches raise an error |
| `partial` | Allows incomplete coverage and reports missing/unmatched IDs |

Aligned annotations retain the base observations, with nulls where metadata is
absent. Partial representation comparison uses the overlapping observations.
Both expose an alignment report with base/incoming counts, coverage and
missing/unmatched IDs. Partial coverage does not silently drop rows from the
source dataset.

Feature metadata uses the same contract:

```python
features_with_metadata = FeatureMatrix(
    dataset.frame.select("activity"),
    observation_ids=dataset.observation_ids,
    metadata=metadata,
    metadata_id_column="id",
    metadata_alignment="partial",
)
print(features_with_metadata.metadata)
print(features_with_metadata.alignment_report)
```

## Missing values and data preservation

Polars `null` and floating-point `NaN` count as missing. Positive and negative
infinity are counted separately as present but non-finite. Numerical analyses
use the eligible finite observations and report sample counts or exclusions;
they do not modify the source dataset. Result tables use nulls for unavailable
values; see [results and exclusions](RESULTS_AND_PROVENANCE.md).

Treat `dataset.frame` and any Polars frame supplied to it as read-only after
construction. The accessor exposes the stored frame, so in-place mutations can
invalidate the recorded schema and IDs. To transform data, create a new table
and a new dataset, preserving or explicitly updating its observation IDs.
Outlier and anomaly analyses flag observations without removing or editing them.
