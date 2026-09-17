# Task 2B — Identity, alignment, FeatureMatrix and annotations in Polars

Branch: `feat/polars-migration`. Task 2A is done (commit `586af3f`): read
`ruddy/data/dataset.py` to see the resulting shape and the adapter comment
style. Read `AGENTS.md` and the "Semánticas" section of
`docs/POLARS_MIGRATION_PLAN.md` first. Do not touch files outside the list
below. No new modules, no new abstractions, no `LazyFrame`.

You have no terminal. Do not run commands. The reviewer runs lint and tests.

## Files to change

- `ruddy/data/validation.py`
- `ruddy/data/feature_matrix.py`
- `ruddy/data/annotations.py`
- `ruddy/data/roles.py` (only: delete `_infer_pandas_kind` once nothing uses it)
- `ruddy/multivariate/permutation.py` (only the `_aligned_labels`-style helper
  around line 76–86; one-line adapter, see §3)
- `ruddy/analysis/engine.py` (only the `align_annotations` call around line
  73; see §3)
- `tests/data/test_alignment.py`, `tests/data/test_feature_matrix.py`,
  `tests/data/test_annotations.py`: keep behaviour tests, adapt input/output
  types, add the tests listed at the end.

`core/io.py` and `results/schemas.py` are **out of scope** (they move with the
CLI in phase 6).

## 1. `validation.py`

- `validate_observation_ids(values, *, source="observations") ->
  tuple[ObservationID, ...]` must not import pandas. Accept any iterable,
  `pl.Series`, `np.ndarray`, `pd.Index`/`pd.Series` (these all iterate to
  Python scalars via `list(values)` or `.to_list()`; use `.to_list()` when
  present, else `list()`). Missing means `None` or a float that is NaN
  (`isinstance(v, float) and v != v`). Duplicates: first-seen order, preview of
  10, `" ..."` suffix. Same exception types/messages as today.
- Delete `_validated_index`.
- `align_annotations(base_ids, annotations, *, id_column=None,
  mode=STRICT) -> tuple[pl.DataFrame, AlignmentReport]`:
  - `annotations` is `pl.DataFrame`, or `pd.DataFrame` (converted with
    `pl.from_pandas(..., include_index=False)`; if `id_column is None` **and**
    the input is pandas, the pandas index is the annotation identity — this is
    the legacy path, keep it with the comment
    `# ponytail: temporary pandas adapter, removed in phase 5`). For a Polars
    input with `id_column is None` raise
    `ValueError("Polars annotations require id_column: Polars has no row index.")`.
  - Validate both ID sets with `validate_observation_ids`. Compute
    `missing`/`unmatched`/`covered` exactly as today (order of base / of
    annotations preserved).
  - Output rows follow `base_ids` order. Implement with a left join of
    `pl.DataFrame({"__ruddy_id": base_ids})` against the annotations on their
    ID column (rename/add a `__ruddy_id` column on the annotation side first),
    `maintain_order="left"`, then drop `__ruddy_id`. If `id_column` is given,
    the output keeps that column, filled with the **base** IDs (as today).
    Missing rows are null in every annotation column.
  - The returned frame has no ID when `id_column is None`; the caller already
    has `base_ids`. Do not attach a synthetic index column.
  - `AlignmentReport` unchanged.

## 2. `feature_matrix.py`

- Accept `pl.DataFrame` in `_normalize_matrix`: all columns must be numeric
  (`dtype.is_numeric()`), else the existing
  `FeatureMatrixValidationError("FeatureMatrix DataFrames must contain only numeric columns.")`;
  `matrix = data.to_numpy()`, feature names = column names, no inferred IDs
  (Polars has no index). Keep the pandas path: numeric check as today, IDs
  from the index **only when the index is not a default RangeIndex**
  (otherwise `None` → generated), with the phase-5 adapter comment.
- Store IDs as `tuple` (`self._observation_ids`). Default when none given:
  `tuple(range(n_rows))`.
- `observation_ids` property keeps returning `pd.Index(self._observation_ids)`
  for now, with the adapter comment (consumers call `.take`/`.to_list`).
- `metadata: pl.DataFrame | None` (no copy; Polars frames are immutable).
  `metadata_id_column` semantics unchanged; pass through to
  `align_annotations`.
- Remove `_validated_index` usage; call `validate_observation_ids` and compare
  `len(...)` to `n_rows`.
- Add `provenance` unchanged.

## 3. Callers of `align_annotations`

- `multivariate/permutation.py` (~line 76–86): the helper builds a pandas
  frame with index = dataset IDs. Replace with
  `annotations = dataset.frame.select(factor).with_columns(pl.Series("__id", dataset.observation_id_tuple))`
  and call `align_annotations(features.observation_ids, annotations, id_column="__id", mode=alignment)`;
  return `aligned[factor].to_pandas()` (the function's declared return is still
  `pd.Series`; add the adapter comment). Import polars there.
- `analysis/engine.py` (~line 73): it only needs the report. Replace the
  `pd.DataFrame(index=...)` argument with
  `pl.DataFrame({"__id": list(features.observation_ids)})` and `id_column="__id"`.
  Import polars; drop the pandas import only if nothing else in the file uses it.

## 4. `annotations.py`

- `AlignedAnnotations` stores `pl.DataFrame` (`data: pl.DataFrame`, no copy).
  `columns` from `self._data.columns`. `to_frame() -> pd.DataFrame` becomes
  `self._data.to_pandas()` with the adapter comment; add `frame -> pl.DataFrame`
  property. In `summary()` for an absent source, `base_count` and
  `missing_ids` come from a new required constructor argument
  `base_ids: tuple[ObservationID, ...]` (an empty Polars frame has no row count
  — see plan "Tablas sin columnas").
- `_resolve_annotation_kinds(frame: pl.DataFrame, overrides)`: use
  `infer_column_kind(frame.schema[column])`; messages unchanged.
- `align_annotation_source`: `annotations: pl.DataFrame | pd.DataFrame | None`.
  Absent → `AlignedAnnotations(data=pl.DataFrame(), base_ids=dataset.observation_id_tuple, ...)`.
  Otherwise call the new `align_annotations` with `dataset.observation_id_tuple`,
  drop `id_column` from the result if present, then roles/kinds/coverage as
  today.
- `attach_annotations`: `base = dataset.frame`; for each available source,
  collision check on column names, then `base = base.hstack(source.frame)`
  (rows are already in base order). Build the new `TabularDataset(base,
  id_column=dataset.id_column, observation_ids=None if dataset.id_column else dataset.observation_id_tuple,
  role_overrides=roles, kind_overrides=kinds)`.
- Then delete `_infer_pandas_kind` from `roles.py` and its import.

## Tests to add/adapt

- `test_alignment.py`: partial alignment with Polars annotations by
  `id_column` returns rows in base order, nulls for missing, report lists
  missing/unmatched in base/annotation order; Polars input without
  `id_column` raises `ValueError`; pandas input with index identity still
  works (legacy path).
- `test_feature_matrix.py`: Polars numeric frame infers feature names and
  generated IDs; Polars frame with a string column is rejected; metadata as a
  Polars frame with `metadata_id_column` aligns partially and
  `matrix.metadata` is a `pl.DataFrame` whose ID column equals the base IDs.
- `test_annotations.py`: existing tests pass with `aligned.frame` /
  `to_frame()`; absent source `summary()` still reports `base_count` and all
  IDs missing.
- `validate_observation_ids([1, None])`, `([1.0, float("nan")])`,
  `(pl.Series(["a", "a"]))` raise the right errors; `(np.array([3, 1, 2]))`
  returns `(3, 1, 2)` as Python ints.

## Acceptance (run by the reviewer)

```bash
uv run task lint
uv run task test
```

533 tests plus your additions must pass. If a test outside the files listed
here fails and you can see why, do not edit it: describe the failure and the
semantic behind it in your summary. Finish with: files touched, tests added,
anything not done and why.
