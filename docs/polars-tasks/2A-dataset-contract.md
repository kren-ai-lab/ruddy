# Task 2A — TabularDataset stores Polars

Branch: `feat/polars-migration`. Read `AGENTS.md` and
`docs/POLARS_MIGRATION_PLAN.md` (sections "Decisiones tomadas" and
"Semánticas que deben fijarse") before editing. Do not touch any file outside
the list below. Do not add features not listed here.

## Files to change

- `ruddy/core/types.py`
- `ruddy/data/roles.py`
- `ruddy/data/validation.py` (only `validate_observation_ids`; leave
  `align_annotations` and `AlignmentReport` untouched)
- `ruddy/data/dataset.py`
- `tests/data/test_dataset.py` (add tests; keep every existing test passing,
  editing an existing test only where this spec changes the contract)

## Target contract for `TabularDataset`

```python
class TabularDataset:
    def __init__(
        self,
        data: pl.DataFrame | pd.DataFrame,
        *,
        id_column: str | None = None,
        observation_ids: Sequence[ObservationID] | None = None,   # NEW
        role_overrides: RoleOverrides | None = None,
        kind_overrides: KindOverrides | None = None,
    ) -> None: ...

    frame: pl.DataFrame            # NEW, read-only property, the stored table
    observation_ids: tuple[ObservationID, ...]   # CHANGED type (see adapters)
    id_column: str | None
    schema: tuple[ColumnSpec, ...]
    provenance: Mapping[str, Any]  # NEW, read-only: {"input_backend": "polars" | "pandas", "id_source": "column" | "argument" | "generated"}
    def role_of(self, column) -> ColumnRole
    def kind_of(self, column) -> ColumnKind
    def columns_with_role(self, *roles) -> tuple[str, ...]
    def columns_with_kind(self, *kinds) -> tuple[str, ...]
```

Rules:

1. **Input.** Accept `pl.DataFrame` as-is (Polars frames are immutable; no
   copy needed). Accept `pd.DataFrame` and convert with `pl.from_pandas(data,
   include_index=False)`. Anything else: `TypeError("data must be a polars or
   pandas DataFrame.")`.
2. **pandas index is never implicit identity.** If a pandas frame arrives
   without `id_column` and without `observation_ids`, and its index is not a
   default `RangeIndex(0..n)`, raise `ValueError` telling the caller to pass
   `observation_ids=frame.index` or `frame.reset_index()`. A default
   RangeIndex falls through to generated IDs.
3. **IDs.** Exactly one source, in this priority: `id_column` (values of that
   column), else `observation_ids` argument, else generated `range(n)`.
   Passing both `id_column` and `observation_ids` raises `ValueError`.
   `observation_ids` length must equal `frame.height`, else `ValueError`.
   Validate with `validate_observation_ids` (see below). Store as a tuple of
   Python scalars (`series.to_list()`); never stringify.
4. **Column names** must be unique strings, same errors as today. Polars
   already guarantees uniqueness; keep the non-string check for the pandas
   path.
5. **Kinds** (`infer_column_kind(dtype: pl.DataType) -> ColumnKind`):
   - `pl.Boolean` → BOOLEAN
   - integer, unsigned integer, float, `pl.Decimal` → NUMERIC
   - `pl.Datetime`, `pl.Date` → DATETIME
   - `pl.String`, `pl.Categorical`, `pl.Enum`, `pl.Utf8` → CATEGORICAL
   - `pl.Duration`, `pl.Time`, `pl.Null`, `pl.Object`, `pl.List`, `pl.Array`,
     `pl.Struct`, `pl.Binary`, anything else → UNKNOWN
   Keep `_validate_kind_override` behaviour and messages verbatim.
   `ColumnSpec.dtype` becomes `str(dtype)` of the Polars dtype (e.g. `"Float64"`,
   `"String"`). This is a documented public change; update the test that
   asserts dtype strings, if any.
6. **Roles**: `resolve_roles`, `resolve_kinds`, `build_schema` take a
   `pl.DataFrame`; logic and error messages unchanged.
7. **`validate_observation_ids(values, *, source="observations") ->
   tuple[ObservationID, ...]`**: accept any sequence, `pl.Series`, `pd.Index`,
   `pd.Series` or `np.ndarray`. Missing means `None` or a float `NaN`. Same
   exception types and messages as today (preview of first 10 duplicates,
   `" ..."` suffix). Return a tuple. It is still called from
   `feature_matrix.py` and `validation.align_annotations`, which expect a
   `pd.Index`; **do not change those callers in this task**, instead add to
   `validation.py`:

   ```python
   def _validated_index(values, *, source="observations") -> pd.Index:
       # ponytail: temporary pandas adapter for feature_matrix/annotations, removed in task 2B
       return pd.Index(validate_observation_ids(values, source=source))
   ```
   and make `align_annotations` and `feature_matrix.py`'s existing calls use
   `_validated_index` (that one-line call-site rename in `feature_matrix.py` is
   allowed).
8. **Temporary pandas adapters on `TabularDataset`**, so the 40+ existing
   consumers keep working unchanged in this task. Each carries the comment
   `# ponytail: temporary pandas adapter, removed in phase 5`:
   - `to_frame() -> pd.DataFrame`: `self.frame.to_pandas()`; if `id_column is
     None`, set the index to `pd.Index(self.observation_ids)` so consumers that
     relied on index identity keep the same IDs.
   - `select(columns) -> pd.DataFrame`: same as `to_frame()` restricted to
     `columns`, preserving the order given.
   - `observation_ids` property returns `pd.Index(self._observation_ids)` for
     now (consumers call `.take`, `.to_list`). Expose the tuple as
     `observation_id_tuple` **only if** you need it in tests; otherwise skip.
   - keep `n_observations`, `n_columns`, `columns`, `__len__` for now (they are
     used by consumers); they will be removed in phase 5.
9. **Types** in `core/types.py`: `ObservationIDs` gains `pl.Series`;
   `FeatureInput` gains `pl.DataFrame`. Import polars at module level.
10. No `LazyFrame`, no `clone()`, no new abstractions, no helper modules.

## Tests to add in `tests/data/test_dataset.py`

- polars input round-trips: `dataset.frame` equals the input
  (`polars.testing.assert_frame_equal`), `provenance["input_backend"] ==
  "polars"`.
- pandas input: `provenance["input_backend"] == "pandas"`; `frame` equals
  `pl.from_pandas(input)`.
- pandas input with a non-default index and no id → `ValueError`.
- `observation_ids=` argument sets identity; wrong length → `ValueError`; with
  `id_column` too → `ValueError`.
- generated IDs are `range(n)` and `provenance["id_source"] == "generated"`.
- kind inference on a polars frame covering Boolean, Int64, Float64, String,
  Categorical, Date, Datetime, Duration (UNKNOWN), Null (UNKNOWN).
- `validate_observation_ids` on a `pl.Series` with a null raises
  `MissingObservationIDError`; with `float("nan")` too.

## Acceptance

```bash
uv run task lint
uv run task test
```

Both must pass. The full suite is 523 tests plus your additions. If a test
outside `tests/data/` fails, do not edit it: stop and report the failure
verbatim, with your diagnosis of which semantic (dtype round-trip, index
identity, category order…) caused it. That report is the deliverable in that
case.

Finish with a short summary: files touched, tests added, anything you could
not do and why.
