"""Error reporting, ``--debug`` and ``--quiet`` behaviour of the CLI shell."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ruddy.cli.main import main
from ruddy.core import RuddyIOError, UnknownColumnError


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "data.csv"
    pd.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]}).to_csv(source, index=False)
    return source


def test_unknown_column_reports_the_available_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["inspect", "profile", str(_source(tmp_path)), "--id-column", "nope"])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "nope" in captured.err
    assert "'id'" in captured.err


def test_missing_input_is_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["inspect", "profile", str(tmp_path / "absent.csv")])

    captured = capsys.readouterr()
    assert code == 2
    assert "absent.csv" in captured.err
    assert "Traceback" not in captured.err


def test_debug_reraises_the_original_error(tmp_path: Path) -> None:
    with pytest.raises(UnknownColumnError):
        main(
            [
                "--debug",
                "inspect",
                "profile",
                str(_source(tmp_path)),
                "--id-column",
                "nope",
            ]
        )


def test_unsupported_extension_lists_the_supported_ones(tmp_path: Path) -> None:
    source = tmp_path / "data.xlsx"
    source.write_bytes(b"")

    with pytest.raises(RuddyIOError, match=r"\.parquet"):
        main(["--debug", "inspect", "profile", str(source)])


def test_quiet_suppresses_the_summary_but_keeps_stdout_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--quiet", "inspect", "profile", str(_source(tmp_path))])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert json.loads(captured.out)


def test_representation_block_runs_with_a_comparison_matrix(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    ids = [f"o{i}" for i in range(24)]
    table = tmp_path / "table.csv"
    pd.DataFrame({"id": ids, "y": rng.normal(size=24)}).to_csv(table, index=False)
    for name in ("x.csv", "z.csv"):
        frame = pd.DataFrame(rng.normal(size=(24, 3)), columns=pd.Index(["a", "b", "c"]))
        frame.insert(0, "id", ids)  # pyrefly: ignore[bad-argument-type]
        frame.to_csv(tmp_path / name, index=False)
    output = tmp_path / "out"

    code = main(
        [
            "analyze",
            "run",
            str(table),
            "--id-column",
            "id",
            "--enable",
            "representation",
            "--feature-input",
            str(tmp_path / "x.csv"),
            "--feature-id-column",
            "id",
            "--comparison-feature-input",
            str(tmp_path / "z.csv"),
            "--comparison-feature-id-column",
            "id",
            "--mantel-permutations",
            "9",
            "--output-dir",
            str(output),
        ]
    )

    assert code == 0
    assert (output / "representation").is_dir()


def test_representation_block_requires_a_comparison_matrix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "analyze",
            "run",
            str(_source(tmp_path)),
            "--id-column",
            "id",
            "--enable",
            "representation",
        ]
    )

    assert code == 2
    assert "--comparison-feature-input" in capsys.readouterr().err
