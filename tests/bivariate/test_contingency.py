from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ruddy import TabularDataset, analyze_contingency_diagnostics


def _dataset() -> TabularDataset:
    rows = []
    counts = {("A", "X"): 30, ("A", "Y"): 10, ("B", "X"): 5, ("B", "Y"): 25}
    for (a, b), n in counts.items():
        rows.extend([(a, b)] * n)
    frame = pl.DataFrame(
        {
            "id": list(range(len(rows))),
            "a": [r[0] for r in rows],
            "b": [r[1] for r in rows],
        }
    )
    return TabularDataset(frame, id_column="id")


def test_cell_contributions_sum_to_chi_square() -> None:
    result = analyze_contingency_diagnostics(_dataset(), p_adjust="none")
    summary = result.summary.row(0, named=True)
    cells = result.cells
    assert cells.get_column("chi_square_contribution").sum() == pytest.approx(summary["chi_square"])
    assert cells.get_column("chi_square_contribution_fraction").sum() == pytest.approx(1.0)


def test_residual_sign_identifies_overrepresentation() -> None:
    result = analyze_contingency_diagnostics(_dataset(), p_adjust="none")
    ax = result.cells.filter((pl.col("level_x") == "A") & (pl.col("level_y") == "X")).row(0, named=True)
    ay = result.cells.filter((pl.col("level_x") == "A") & (pl.col("level_y") == "Y")).row(0, named=True)
    assert ax["pearson_residual"] > 0
    assert ay["pearson_residual"] < 0
    assert np.isfinite(ax["standardized_residual"])


def test_contingency_respects_level_guard() -> None:
    frame = pl.DataFrame({"id": list(range(12)), "a": [f"a{i}" for i in range(12)], "b": ["x", "y"] * 6})
    ds = TabularDataset(frame, id_column="id")
    result = analyze_contingency_diagnostics(ds, max_category_levels=5)
    assert result.summary.row(0, named=True)["status"] == "skipped"
    assert result.cells.height == 0
