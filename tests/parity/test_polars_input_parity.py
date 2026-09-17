"""Pin which third-party entry points accept polars input and match pandas results.

These are the integration boundaries the polars migration relies on. Patsy is
the one dependency that does not accept polars, so it is pinned as failing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
from patsy import PatsyError, dmatrices  # pyrefly: ignore[missing-module-attribute]
from scipy import stats
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import mixedlm, ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera


@pytest.fixture(scope="module")
def frames() -> tuple[pl.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    polars_frame = pl.DataFrame(
        {
            "y": rng.normal(size=60),
            "x": rng.normal(size=60),
            "z": rng.normal(size=60),
            "g": np.repeat(["a", "b", "c"], 20),
            "G": np.tile(["s1", "s2", "s3", "s4"], 15),
        }
    )
    return polars_frame, polars_frame.to_pandas()


def test_sklearn_accepts_polars(frames) -> None:
    pl_frame, pd_frame = frames
    pl_num, pd_num = pl_frame.select("y", "x", "z"), pd_frame[["y", "x", "z"]]
    np.testing.assert_allclose(PCA(2).fit_transform(pl_num), PCA(2).fit_transform(pd_num))
    np.testing.assert_allclose(StandardScaler().fit_transform(pl_num), StandardScaler().fit_transform(pd_num))
    np.testing.assert_array_equal(
        IsolationForest(random_state=0).fit_predict(pl_num),
        IsolationForest(random_state=0).fit_predict(pd_num),
    )
    np.testing.assert_allclose(
        MinCovDet(random_state=0).fit(pl_num).covariance_,
        MinCovDet(random_state=0).fit(pd_num).covariance_,
    )


def test_scipy_accepts_polars_series(frames) -> None:
    pl_frame, pd_frame = frames
    assert stats.pearsonr(pl_frame["x"], pl_frame["y"]) == stats.pearsonr(pd_frame["x"], pd_frame["y"])
    assert stats.shapiro(pl_frame["x"]) == stats.shapiro(pd_frame["x"])


def test_statsmodels_accepts_polars(frames) -> None:
    pl_frame, pd_frame = frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl_ols, pd_ols = ols("y ~ C(g) + x", data=pl_frame).fit(), ols("y ~ C(g) + x", data=pd_frame).fit()
        pl_mixed = mixedlm("y ~ x", pl_frame, groups=pl_frame["G"]).fit()
        pd_mixed = mixedlm("y ~ x", pd_frame, groups=pd_frame["G"]).fit()
        pl_manova = MANOVA.from_formula("y + x ~ C(g)", data=pl_frame).mv_test().results["C(g)"]["stat"]
        pd_manova = MANOVA.from_formula("y + x ~ C(g)", data=pd_frame).mv_test().results["C(g)"]["stat"]

    pd.testing.assert_series_equal(pl_ols.params, pd_ols.params)
    pd.testing.assert_frame_equal(anova_lm(pl_ols, typ=2), anova_lm(pd_ols, typ=2))
    pd.testing.assert_series_equal(pl_mixed.params, pd_mixed.params)
    np.testing.assert_allclose(pl_manova.to_numpy(float), pd_manova.to_numpy(float))
    assert normal_ad(pl_frame["x"]) == normal_ad(pd_frame["x"])
    assert jarque_bera(pl_frame["x"]) == jarque_bera(pd_frame["x"])
    pl_bp = het_breuschpagan(pl_ols.resid, pl_ols.model.exog)
    assert pl_bp == het_breuschpagan(pd_ols.resid, pd_ols.model.exog)
    pl_num, pd_num = pl_frame.select("y", "x", "z"), pd_frame[["y", "x", "z"]]
    assert variance_inflation_factor(pl_num, 0) == variance_inflation_factor(pd_num, 0)


def test_patsy_rejects_polars_and_needs_explicit_to_pandas(frames) -> None:
    pl_frame, _ = frames
    with pytest.raises(PatsyError):
        dmatrices("y ~ C(g) + x", data=pl_frame, return_type="dataframe")
    _, design = dmatrices("y ~ C(g) + x", data=pl_frame.to_pandas(), return_type="dataframe")
    assert design.shape == (60, 4)
