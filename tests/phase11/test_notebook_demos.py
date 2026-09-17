from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "examples" / "notebooks"
DATA = ROOT / "examples" / "data"
EXPECTED = [
    "01_profiling_univariate.ipynb",
    "02_bivariate_dependence.ipynb",
    "03_groups_posthoc_intervals.ipynb",
    "04_factorial_marginal_mixed.ipynb",
    "05_feature_spaces_projections.ipynb",
    "06_multivariate_permanova.ipynb",
    "07_representation_comparison.ipynb",
    "08_compositional.ipynb",
    "09_outliers_anomaly.ipynb",
    "10_bayesian_eda.ipynb",
    "11_numerical_representation_framework.ipynb",
    "12_end_to_end_generic_eda.ipynb",
    "13_visualization_gallery.ipynb",
]
INTERACTIVE = {
    "02_bivariate_dependence.ipynb",
    "05_feature_spaces_projections.ipynb",
    "07_representation_comparison.ipynb",
    "11_numerical_representation_framework.ipynb",
}


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _outputs(notebook: dict):
    return [
        output
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        for output in cell.get("outputs", [])
    ]


def test_expected_notebook_catalog_exists() -> None:
    assert [path.name for path in sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))] == EXPECTED


def test_notebooks_are_executed_without_error_outputs() -> None:
    for name in EXPECTED:
        notebook = _load_notebook(NOTEBOOKS / name)
        assert notebook.get("metadata", {}).get("ruddy_demo", {}).get("phase") == "11-strong"
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        assert code_cells, name
        assert all(cell.get("execution_count") is not None for cell in code_cells), name
        errors = [output for output in _outputs(notebook) if output.get("output_type") == "error"]
        assert not errors, f"{name}: {errors}"


def test_every_notebook_is_visually_rich() -> None:
    for name in EXPECTED:
        notebook = _load_notebook(NOTEBOOKS / name)
        images = [
            output
            for output in _outputs(notebook)
            if output.get("output_type") in {"display_data", "execute_result"}
            and "image/png" in output.get("data", {})
        ]
        minimum = 8 if name == "13_visualization_gallery.ipynb" else 4
        assert len(images) >= minimum, f"{name} contains only {len(images)} static figures"


def test_selected_notebooks_include_interactive_plotly_views() -> None:
    for name in INTERACTIVE:
        notebook = _load_notebook(NOTEBOOKS / name)
        plotly = [
            output
            for output in _outputs(notebook)
            if output.get("output_type") in {"display_data", "execute_result"}
            and "application/vnd.plotly.v1+json" in output.get("data", {})
        ]
        assert plotly, f"{name} does not contain an executed interactive Plotly view"


def test_notebooks_use_ruddy_results_and_comparative_views() -> None:
    for name in EXPECTED:
        notebook = _load_notebook(NOTEBOOKS / name)
        source = "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", ""), list)
            else cell.get("source", "")
            for cell in notebook["cells"]
        )
        assert "from ruddy import" in source, name
        assert notebook.get("metadata", {}).get("ruddy_demo", {}).get("comparative") is True


def test_demo_feature_spaces_share_observation_identity() -> None:
    ids = pd.read_csv(DATA / "tabular_demo.csv", usecols=["id"])["id"].tolist()
    for name in (
        "representation_a.csv",
        "representation_b.csv",
        "representation_c.csv",
        "compositional_demo.csv",
    ):
        assert pd.read_csv(DATA / name, usecols=["id"])["id"].tolist() == ids


def test_visualization_dependencies_do_not_leak_into_core() -> None:
    forbidden = ("import matplotlib", "from matplotlib", "import plotly", "from plotly")
    package = ROOT / "ruddy"
    assert package.is_dir(), package
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
