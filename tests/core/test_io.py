from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ruddy.core.exceptions import RuddyIOError
from ruddy.core.io import read_table, write_table


def test_csv_round_trip(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "table.csv"
    written = write_table(df, path)
    assert written == path
    assert path.exists()

    result = read_table(path)
    assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result, df)


def test_tsv_round_trip(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "table.tsv"
    written = write_table(df, path)
    assert written == path

    result = read_table(path)
    assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result, df)


def test_txt_tsv_round_trip(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "table.txt"
    written = write_table(df, path)
    assert written == path

    result = read_table(path)
    assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result, df)


def test_parquet_round_trip(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "table.parquet"
    written = write_table(df, path)
    assert written == path

    result = read_table(path)
    assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result, df)


def test_pandas_written_parquet_with_index_retains_column(tmp_path: Path) -> None:
    df_pd = pd.DataFrame(
        {"value": [10.0, 20.0]},
        index=pd.Index(["row1", "row2"], name="sample_id"),
    )
    parquet_path = tmp_path / "pandas_indexed.parquet"
    df_pd.to_parquet(parquet_path)

    result = read_table(parquet_path)
    assert isinstance(result, pl.DataFrame)
    assert "sample_id" in result.columns
    assert "value" in result.columns
    assert result["sample_id"].to_list() == ["row1", "row2"]
    assert result["value"].to_list() == [10.0, 20.0]


def test_read_table_nonexistent_file() -> None:
    with pytest.raises(RuddyIOError, match="not found"):
        read_table("nonexistent_path_ruddy_test.csv")


def test_read_table_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "file.xlsx"
    path.touch()
    with pytest.raises(RuddyIOError, match="Unsupported file extension"):
        read_table(path)


def test_write_table_unsupported_extension(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(RuddyIOError, match="Unsupported file extension"):
        write_table(df, tmp_path / "file.xlsx")
