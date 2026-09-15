from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ruddy.cli.main import main


def _write_fixture(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x": [1.0, 2.0, np.nan, 4.0],
            "group": ["A", "A", "B", "B"],
        }
    ).to_csv(path, index=False)
    return path


def test_profile_cli_writes_structured_outputs(tmp_path) -> None:
    source = _write_fixture(tmp_path)
    output = tmp_path / "profile"
    assert (
        main(
            [
                "inspect",
                "profile",
                str(source),
                "--id-column",
                "id",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert {path.name for path in output.iterdir()} == {
        "overview.json",
        "provenance.json",
        "columns.csv",
        "missingness.csv",
        "pairwise_completeness.csv",
        "missingness_patterns.csv",
    }
    overview = json.loads((output / "overview.json").read_text())
    assert overview["n_records"] == 4


def test_univariate_cli_writes_scientific_tables(tmp_path) -> None:
    source = _write_fixture(tmp_path)
    output = tmp_path / "univariate"
    assert (
        main(
            [
                "inspect",
                "univariate",
                str(source),
                "--id-column",
                "id",
                "--factor",
                "group",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "numeric_statistics.csv").is_file()
    assert (output / "categorical_statistics.csv").is_file()
    assert (output / "categorical_frequencies.csv").is_file()
    assert (output / "datetime_statistics.csv").is_file()
