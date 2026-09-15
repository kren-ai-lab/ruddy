from __future__ import annotations

import pandas as pd

from ruddy.cli.main import main


def _base_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"o{i}" for i in range(1, 9)],
            "activity": [1.0, 1.1, 1.2, 1.3, 5.0, 5.1, 5.2, 5.3],
            "family": ["A"] * 4 + ["B"] * 4,
        }
    )


def test_groups_cli_writes_structured_outputs(tmp_path) -> None:
    source = tmp_path / "data.csv"
    output = tmp_path / "results"
    _base_table().to_csv(source, index=False)

    code = main(
        [
            "analyze",
            "groups",
            str(source),
            "--id-column",
            "id",
            "--response",
            "activity",
            "--group",
            "family",
            "--min-group-n",
            "2",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    expected = {
        "response_catalog.csv",
        "group_coverage.csv",
        "grouped_numeric_summaries.csv",
        "grouped_categorical_summaries.csv",
        "grouped_numeric_comparisons.csv",
        "grouped_categorical_comparisons.csv",
        "annotation_coverage.csv",
        "grouped_provenance.json",
    }
    assert expected.issubset({path.name for path in output.iterdir()})


def test_groups_cli_supports_partial_external_annotations(tmp_path) -> None:
    source = tmp_path / "data.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "results"
    _base_table().to_csv(source, index=False)
    pd.DataFrame(
        {
            "id": ["o1", "o2", "o5", "o6", "outside"],
            "cohort": ["C1", "C1", "C2", "C2", "C9"],
        }
    ).to_csv(annotations, index=False)

    code = main(
        [
            "analyze",
            "groups",
            str(source),
            "--id-column",
            "id",
            "--response",
            "activity",
            "--group",
            "cohort",
            "--annotation-file",
            str(annotations),
            "--annotation-id-column",
            "id",
            "--annotation-alignment",
            "partial",
            "--annotation-factor",
            "cohort",
            "--min-group-n",
            "2",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    coverage = pd.read_csv(output / "annotation_coverage.csv")
    assert coverage.loc[0, "coverage"] == "partial"
    assert coverage.loc[0, "covered_count"] == 4
