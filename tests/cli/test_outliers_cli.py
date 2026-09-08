from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ruddy.cli.main import main


def _input(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(8)],
            "x": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 100.0],
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    source = path / "data.csv"
    frame.to_csv(source, index=False)
    return source


def test_outliers_cli_writes_structured_outputs(tmp_path: Path) -> None:
    source = _input(tmp_path)
    output = tmp_path / "results"
    code = main(
        [
            "outliers",
            str(source),
            "--id-column",
            "id",
            "--factor",
            "group",
            "--include-flags",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert (output / "outlier_summaries.csv").is_file()
    assert (output / "numeric_quality.csv").is_file()
    assert (output / "outlier_flags.csv").is_file()
    provenance = json.loads((output / "outlier_provenance.json").read_text())
    assert provenance["analysis"] == "outliers"
    summaries = pd.read_csv(output / "outlier_summaries.csv")
    assert set(summaries["column"]) == {"x"}


def test_outliers_cli_help_is_registered(capsys) -> None:
    try:
        main(["outliers", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "outlier" in capsys.readouterr().out.lower()
