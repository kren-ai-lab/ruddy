from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ruddy.cli.main import main


def test_multivariate_cli_writes_structured_outputs(tmp_path: Path):
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(50, 4)), columns=["f1", "f2", "f3", "f4"])
    frame.insert(0, "id", [f"o{i}" for i in range(len(frame))])
    source = tmp_path / "features.csv"
    target = tmp_path / "results"
    frame.to_csv(source, index=False)

    code = main([
        "multivariate", str(source), "--id-column", "id", "--output-dir", str(target)
    ])
    assert code == 0
    expected = {
        "covariance_matrix.csv",
        "pearson_matrix.csv",
        "spearman_matrix.csv",
        "pairwise_counts.csv",
        "covariance_feature_diagnostics.csv",
        "covariance_condition_spectrum.csv",
        "covariance_summary.json",
        "collinearity.csv",
        "collinearity_condition_spectrum.csv",
        "collinearity_summary.json",
        "mahalanobis_methods.csv",
        "mahalanobis_distances.csv",
        "multivariate_provenance.json",
    }
    assert expected <= {path.name for path in target.iterdir()}
    payload = json.loads((target / "multivariate_provenance.json").read_text())
    assert payload["status"] == "ok"


def test_manova_cli_writes_structured_outputs(tmp_path: Path):
    rng = np.random.default_rng(6)
    n = 48
    group = np.repeat(["A", "B"], n // 2)
    frame = pd.DataFrame({
        "id": [f"o{i}" for i in range(n)],
        "y1": rng.normal(size=n) + (group == "B") * 0.8,
        "y2": rng.normal(size=n) + (group == "B") * 0.4,
        "group": group,
    })
    source = tmp_path / "table.csv"
    target = tmp_path / "manova"
    frame.to_csv(source, index=False)

    code = main([
        "manova", str(source), "--id-column", "id",
        "--response", "y1", "--response", "y2",
        "--factor", "group", "--output-dir", str(target),
    ])
    assert code == 0
    assert (target / "manova_tests.csv").exists()
    assert (target / "manova_factor_levels.csv").exists()
    assert (target / "manova_model_summary.json").exists()
    assert (target / "manova_provenance.json").exists()
