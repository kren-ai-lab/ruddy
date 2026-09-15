from __future__ import annotations

import pandas as pd
import pytest

from ruddy import ColumnKind, ColumnRole, TabularDataset
from ruddy.core.exceptions import (
    DuplicateObservationIDError,
    KindConflictError,
    MissingObservationIDError,
    RoleConflictError,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "value": [1.0, 2.5, 3.0],
            "numeric_string": ["1", "2", "3"],
            "group": ["x", "x", "y"],
            "flag": [True, False, True],
        }
    )


def test_tabular_dataset_does_not_mutate_input() -> None:
    frame = make_frame()
    original = frame.copy(deep=True)

    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={"group": "factor", "numeric_string": "annotation"},
    )

    pd.testing.assert_frame_equal(frame, original)

    external = dataset.to_frame()
    external.loc[0, "value"] = 999.0
    pd.testing.assert_frame_equal(dataset.to_frame(), original)


def test_duplicate_ids_fail_explicitly() -> None:
    frame = make_frame()
    frame.loc[2, "id"] = "a"

    with pytest.raises(DuplicateObservationIDError):
        TabularDataset(frame, id_column="id")


def test_missing_ids_fail_explicitly() -> None:
    frame = make_frame()
    frame.loc[1, "id"] = None

    with pytest.raises(MissingObservationIDError):
        TabularDataset(frame, id_column="id")


def test_roles_and_kinds_are_resolved_deterministically() -> None:
    dataset = TabularDataset(
        make_frame(),
        id_column="id",
        role_overrides={"group": ColumnRole.FACTOR, "value": ColumnRole.RESPONSE},
        kind_overrides={"value": ColumnKind.NUMERIC},
    )

    assert dataset.role_of("id") is ColumnRole.IDENTIFIER
    assert dataset.role_of("group") is ColumnRole.FACTOR
    assert dataset.role_of("value") is ColumnRole.RESPONSE
    assert dataset.kind_of("value") is ColumnKind.NUMERIC
    assert dataset.kind_of("numeric_string") is ColumnKind.CATEGORICAL
    assert dataset.kind_of("flag") is ColumnKind.BOOLEAN


def test_identifier_role_conflict_fails_at_construction() -> None:
    with pytest.raises(RoleConflictError):
        TabularDataset(
            make_frame(),
            id_column="id",
            role_overrides={"id": "annotation"},
        )


def test_second_identifier_role_conflicts_with_id_column() -> None:
    with pytest.raises(RoleConflictError):
        TabularDataset(
            make_frame(),
            id_column="id",
            role_overrides={"group": "identifier"},
        )


def test_string_to_numeric_override_is_rejected_without_coercion() -> None:
    with pytest.raises(KindConflictError, match="does not silently coerce strings"):
        TabularDataset(
            make_frame(),
            id_column="id",
            kind_overrides={"numeric_string": "numeric"},
        )


def test_numeric_column_can_be_explicitly_treated_as_categorical() -> None:
    dataset = TabularDataset(
        make_frame(),
        id_column="id",
        kind_overrides={"value": "categorical"},
    )

    assert dataset.kind_of("value") is ColumnKind.CATEGORICAL


def test_duplicate_column_names_fail_explicitly() -> None:
    frame = pd.DataFrame([[1, 2]], columns=pd.Index(["x", "x"]))
    with pytest.raises(ValueError, match="column names must be unique"):
        TabularDataset(frame)


def test_non_string_column_names_are_rejected_for_stable_schema() -> None:
    frame = pd.DataFrame([[1, 2]], columns=pd.Index([0, 1]))
    with pytest.raises(TypeError, match="string column names"):
        TabularDataset(frame)
