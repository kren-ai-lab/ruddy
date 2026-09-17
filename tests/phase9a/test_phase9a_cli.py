from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ruddy.cli.main import main


def _write_dataset(path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 30
    g = np.repeat(["A", "B"], 15)
    pd.DataFrame(
        {
            "id": range(n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "z": rng.normal(size=n),
            "group": g,
            "cat": ["X", "Y"] * 15,
        }
    ).to_csv(path, index=False)


def test_diagnostics_cli(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    out = tmp_path / "diag"
    _write_dataset(source)
    code = main(
        [
            "inspect",
            "diagnostics",
            str(source),
            "--id-column",
            "id",
            "--response",
            "x",
            "--factor",
            "group",
            "--group",
            "group",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "normality_diagnostics.csv").exists()
    assert (out / "dispersion_diagnostics.csv").exists()


def test_dependence_cli(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    out = tmp_path / "dep"
    _write_dataset(source)
    code = main(
        [
            "analyze",
            "dependence",
            str(source),
            "--id-column",
            "id",
            "--covariate",
            "z",
            "--partial-covariate",
            "z",
            "--dependence-permutations",
            "9",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "partial_correlations.csv").exists()
    assert (out / "distance_correlations.csv").exists()
    assert (out / "mutual_information.csv").exists()


def test_contingency_cli(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    out = tmp_path / "ct"
    _write_dataset(source)
    code = main(
        [
            "analyze",
            "contingency",
            str(source),
            "--id-column",
            "id",
            "--factor",
            "group",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "contingency_cells.csv").exists()


def test_intervals_cli(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    out = tmp_path / "ci"
    _write_dataset(source)
    code = main(
        [
            "analyze",
            "intervals",
            str(source),
            "--id-column",
            "id",
            "--factor",
            "group",
            "--bootstrap-resamples",
            "100",
            "--bootstrap-method",
            "percentile",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "mean_intervals.csv").exists()
    assert (out / "effect_size_intervals.csv").exists()


def test_unified_cli_writes_phase9a_subdirectories(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    out = tmp_path / "all"
    _write_dataset(source)
    code = main(
        [
            "pipeline",
            str(source),
            "--id-column",
            "id",
            "--response",
            "x",
            "--factor",
            "group",
            "--group",
            "group",
            "--covariate",
            "z",
            "--partial-covariate",
            "z",
            "--dependence-permutations",
            "9",
            "--bootstrap-resamples",
            "100",
            "--bootstrap-method",
            "percentile",
            "--enable",
            "diagnostics",
            "--enable",
            "dependence",
            "--enable",
            "contingency",
            "--enable",
            "intervals",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    for name in ("diagnostics", "dependence", "contingency", "intervals"):
        assert (out / name).is_dir()
    assert (out / "analysis_summary.json").exists()
