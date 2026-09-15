from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ruddy.cli.main import main


def _write_dataset(path: Path) -> None:
    rng = np.random.default_rng(41)
    rows = []
    for a in ("A0", "A1"):
        for b in ("B0", "B1"):
            for index in range(12):
                x = rng.normal()
                y = (
                    1.0
                    + (a == "A1") * 0.8
                    + (b == "B1") * 0.5
                    + 0.4 * x
                    + rng.normal(scale=0.5)
                )
                rows.append((f"{a}_{b}_{index}", y, a, b, x))
    pd.DataFrame(rows, columns=("id", "y", "a", "b", "x")).to_csv(path, index=False)


def test_factorial_cli_formula_writes_structured_outputs(tmp_path: Path):
    source = tmp_path / "data.csv"
    target = tmp_path / "factorial"
    _write_dataset(source)
    code = main(
        [
            "model",
            "factorial",
            str(source),
            "--id-column",
            "id",
            "--response",
            "y",
            "--factor",
            "a",
            "--factor",
            "b",
            "--covariate",
            "x",
            "--formula",
            "y ~ a * b + x",
            "--ss-type",
            "3",
            "--output-dir",
            str(target),
        ]
    )
    assert code == 0
    expected = {
        "factorial_design_terms.csv",
        "factorial_effects.csv",
        "factorial_coefficients.csv",
        "factorial_diagnostics.csv",
        "factorial_observation_diagnostics.csv",
        "factorial_cells.csv",
        "factorial_model_summary.json",
        "factorial_provenance.json",
    }
    assert expected <= {path.name for path in target.iterdir()}
    payload = json.loads((target / "factorial_provenance.json").read_text())
    assert payload["status"] == "ok"


def test_factorial_cli_explicit_interaction(tmp_path: Path):
    source = tmp_path / "data.csv"
    target = tmp_path / "factorial"
    _write_dataset(source)
    code = main(
        [
            "model",
            "factorial",
            str(source),
            "--id-column",
            "id",
            "--response",
            "y",
            "--factor",
            "a",
            "--factor",
            "b",
            "--interaction",
            "a:b",
            "--output-dir",
            str(target),
        ]
    )
    assert code == 0
    effects = pd.read_csv(target / "factorial_effects.csv")
    assert "a:b" in set(effects["term"])


def test_factorial_cli_requires_one_response_without_formula(tmp_path: Path):
    source = tmp_path / "data.csv"
    _write_dataset(source)
    code = main(
        ["model", "factorial", str(source), "--id-column", "id", "--factor", "a"]
    )
    assert code == 2, "CLI should reject a factorial model without one response."
