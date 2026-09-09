from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ruddy.cli.main import main


def _write_data(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(22)
    n = 32
    group = np.repeat(["A", "B"], n // 2)
    x = rng.normal(size=n)
    y = x + (group == "B") * 1.2 + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame({
        "id": [f"s{i}" for i in range(n)],
        "x": x,
        "z": rng.normal(size=n),
        "y": y,
        "group": group,
    })
    frame.to_csv(path, index=False)
    return frame


def test_analyze_cli_default_is_descriptive_only(tmp_path, capsys):
    source = tmp_path / "data.csv"
    _write_data(source)
    assert main(["analyze", str(source), "--id-column", "id"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed_blocks"] == ["profiling", "univariate"]
    assert set(payload["components"]) == {"profiling", "univariate"}


def test_analyze_cli_explicit_blocks_write_separate_artifacts(tmp_path):
    source = tmp_path / "data.csv"
    target = tmp_path / "results"
    _write_data(source)
    assert main([
        "analyze", str(source), "--id-column", "id",
        "--response", "y", "--factor", "group", "--group", "group",
        "--enable", "profiling", "--enable", "groups", "--enable", "outliers",
        "--output-dir", str(target),
    ]) == 0
    assert (target / "analysis_summary.json").is_file()
    assert (target / "profiling" / "overview.json").is_file()
    assert (target / "groups" / "grouped_numeric_summaries.csv").is_file()
    assert (target / "outliers" / "outlier_summaries.csv").is_file()
    assert not (target / "univariate").exists()


def test_analyze_cli_pca_uses_explicit_feature_columns(tmp_path):
    source = tmp_path / "data.csv"
    target = tmp_path / "results"
    _write_data(source)
    assert main([
        "analyze", str(source), "--id-column", "id",
        "--enable", "pca", "--feature-column", "x", "--feature-column", "z",
        "--projection-n-components", "2", "--projection-scaling", "standard",
        "--output-dir", str(target),
    ]) == 0
    assert (target / "pca" / "pca_scores.csv").is_file()
    summary = json.loads((target / "analysis_summary.json").read_text())
    assert summary["executed_blocks"] == ["pca"]
    assert summary["feature_alignment"]["complete"] is True


def test_analyze_cli_factorial_formula(tmp_path):
    source = tmp_path / "data.csv"
    target = tmp_path / "results"
    _write_data(source)
    assert main([
        "analyze", str(source), "--id-column", "id",
        "--response", "y", "--factor", "group", "--covariate", "x",
        "--enable", "factorial", "--factorial-formula", "y ~ group + x",
        "--ss-type", "3", "--output-dir", str(target),
    ]) == 0
    assert (target / "factorial" / "factorial_effects.csv").is_file()


def test_analyze_cli_bivariate_only_does_not_write_descriptive_dirs(tmp_path):
    source = tmp_path / "data.csv"
    target = tmp_path / "results"
    _write_data(source)
    assert main([
        "analyze", str(source), "--id-column", "id",
        "--enable", "bivariate", "--output-dir", str(target),
    ]) == 0
    assert (target / "bivariate" / "correlations.csv").is_file()
    assert not (target / "profiling").exists()
    assert not (target / "univariate").exists()
