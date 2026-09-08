from __future__ import annotations

import pandas as pd

from ruddy.cli.main import main


def test_bivariate_cli_writes_structured_outputs(tmp_path) -> None:
    source = tmp_path / "data.csv"
    output = tmp_path / "out"
    pd.DataFrame(
        {
            "id": [f"s{i}" for i in range(8)],
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [2, 3, 4, 5, 7, 8, 9, 10],
            "group": ["A"] * 4 + ["B"] * 4,
            "flag": [True, False] * 4,
        }
    ).to_csv(source, index=False)
    code = main(
        [
            "bivariate",
            str(source),
            "--id-column",
            "id",
            "--factor",
            "group",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert (output / "correlations.csv").exists()
    assert (output / "comparisons.csv").exists()
    assert (output / "categorical_associations.csv").exists()
    assert (output / "bivariate_provenance.json").exists()
