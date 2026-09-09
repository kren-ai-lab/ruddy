from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ruddy import TabularDataset, analyze_distribution_diagnostics
from ruddy.univariate.diagnostics import summarize_normality_diagnostics


def _dataset() -> TabularDataset:
    rng = np.random.default_rng(12)
    n = 80
    frame = pd.DataFrame({
        "id": [f"o{i}" for i in range(n)],
        "normal": rng.normal(size=n),
        "skewed": rng.exponential(size=n),
        "constant": np.ones(n),
        "group": np.repeat(["A","B"], n//2),
    })
    frame.loc[frame.group.eq("B"), "normal"] *= 2.5
    return TabularDataset(frame, id_column="id", role_overrides={"group":"factor"})


def test_normality_methods_are_standalone() -> None:
    result = analyze_distribution_diagnostics(_dataset())
    assert set(result.normality.method) == {"shapiro","dagostino","anderson_darling"}
    assert len(result.dispersion) == 0
    assert result.provenance.parameters["automatic_test_selection"] is False


def test_constant_is_degenerate() -> None:
    table = summarize_normality_diagnostics(_dataset(), columns=("constant",))
    assert set(table.status) == {"degenerate"}
    assert set(table.reason) == {"constant"}


def test_shapiro_sample_limit_is_explicit() -> None:
    table = summarize_normality_diagnostics(_dataset(), methods=("shapiro",), max_shapiro_n=20)
    rows = table[table.column.eq("normal")]
    assert rows.iloc[0].status == "skipped"
    assert rows.iloc[0].reason == "sample_size_exceeds_shapiro_limit"


def test_group_dispersion_matches_scipy() -> None:
    ds = _dataset()
    result = analyze_distribution_diagnostics(ds, responses=("normal",), groups=("group",), p_adjust="none")
    brown = result.dispersion[result.dispersion.method.eq("brown_forsythe")].iloc[0]
    frame = ds.to_frame(); a = frame.loc[frame.group.eq("A"),"normal"]; b = frame.loc[frame.group.eq("B"),"normal"]
    reference = stats.levene(a, b, center="median")
    assert brown.statistic == pytest.approx(reference.statistic)
    assert brown.p_value == pytest.approx(reference.pvalue)


def test_numeric_factor_is_not_normality_variable() -> None:
    frame = pd.DataFrame({"id":range(20), "x":np.arange(20.0), "batch":np.tile([0,1],10)})
    ds = TabularDataset(frame, id_column="id", role_overrides={"batch":"factor"})
    table = summarize_normality_diagnostics(ds)
    assert "batch" not in set(table.column)
