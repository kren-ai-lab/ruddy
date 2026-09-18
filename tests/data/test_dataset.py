from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ruddy import ColumnKind, ColumnRole, TabularDataset
from ruddy.core.exceptions import (
    DuplicateObservationIDError,
    KindConflictError,
    MissingObservationIDError,
    RoleConflictError,
)
from ruddy.data.validation import validate_observation_ids


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

    assert dataset.frame.shape == (3, 5)
    frame.loc[0, "value"] = 999.0
    assert dataset.frame["value"][0] == 1.0


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


def test_polars_input_round_trips() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    dataset = TabularDataset(df)
    assert_frame_equal(dataset.frame, df)
    assert dataset.provenance["input_backend"] == "polars"


def test_pandas_input() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    dataset = TabularDataset(df)
    assert dataset.provenance["input_backend"] == "pandas"
    assert_frame_equal(dataset.frame, pl.from_pandas(df, include_index=False))


def test_pandas_input_non_default_index_no_id() -> None:
    df = pd.DataFrame({"a": [1, 2]}, index=[1, 2])
    with pytest.raises(ValueError, match="pandas DataFrame has a non-default index"):
        TabularDataset(df)


def test_observation_ids_argument_sets_identity() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    dataset = TabularDataset(df, observation_ids=["x", "y"])
    assert dataset.observation_ids == ("x", "y")
    assert dataset.provenance["id_source"] == "argument"


def test_observation_ids_wrong_length() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="observation_ids length must equal"):
        TabularDataset(df, observation_ids=["x"])


def test_observation_ids_with_id_column_fails() -> None:
    df = pl.DataFrame({"id": ["x", "y"], "a": [1, 2]})
    with pytest.raises(ValueError, match="Cannot pass both id_column and observation_ids"):
        TabularDataset(df, id_column="id", observation_ids=["x", "y"])


def test_generated_ids() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    dataset = TabularDataset(df)
    assert dataset.observation_ids == (0, 1)
    assert dataset.provenance["id_source"] == "generated"


def test_kind_inference_polars_frame() -> None:
    df = pl.DataFrame(
        {
            "bool": pl.Series([True], dtype=pl.Boolean),
            "int": pl.Series([1], dtype=pl.Int64),
            "float": pl.Series([1.5], dtype=pl.Float64),
            "str": pl.Series(["a"], dtype=pl.String),
            "cat": pl.Series(["a"], dtype=pl.Categorical),
            "date": pl.Series(["2020-01-01"]).str.strptime(pl.Date),
            "datetime": pl.Series(["2020-01-01T00:00:00"]).str.strptime(pl.Datetime),
            "duration": pl.Series([100], dtype=pl.Duration),
            "null": pl.Series([None], dtype=pl.Null),
        }
    )
    dataset = TabularDataset(df)
    assert dataset.kind_of("bool") is ColumnKind.BOOLEAN
    assert dataset.kind_of("int") is ColumnKind.NUMERIC
    assert dataset.kind_of("float") is ColumnKind.NUMERIC
    assert dataset.kind_of("str") is ColumnKind.CATEGORICAL
    assert dataset.kind_of("cat") is ColumnKind.CATEGORICAL
    assert dataset.kind_of("date") is ColumnKind.DATETIME
    assert dataset.kind_of("datetime") is ColumnKind.DATETIME
    assert dataset.kind_of("duration") is ColumnKind.UNKNOWN
    assert dataset.kind_of("null") is ColumnKind.UNKNOWN


def test_validate_observation_ids_polars_series_with_null_and_nan() -> None:
    with pytest.raises(MissingObservationIDError):
        validate_observation_ids(pl.Series(["a", None]))
    with pytest.raises(MissingObservationIDError):
        validate_observation_ids(pl.Series([1.0, float("nan")]))


def test_deleted_dataset_contract_members() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    dataset = TabularDataset(df)
    with pytest.raises(AttributeError):
        _ = dataset.to_frame  # pyrefly: ignore[missing-attribute]
    with pytest.raises(AttributeError):
        _ = dataset.select  # pyrefly: ignore[missing-attribute]
    with pytest.raises(AttributeError):
        _ = dataset.n_observations  # pyrefly: ignore[missing-attribute]
    with pytest.raises(AttributeError):
        _ = dataset.n_columns  # pyrefly: ignore[missing-attribute]
    with pytest.raises(AttributeError):
        _ = dataset.columns  # pyrefly: ignore[missing-attribute]
    with pytest.raises(AttributeError):
        _ = dataset.observation_id_tuple  # pyrefly: ignore[missing-attribute]
    with pytest.raises(TypeError):
        _ = len(dataset)  # pyrefly: ignore[bad-argument-type]
