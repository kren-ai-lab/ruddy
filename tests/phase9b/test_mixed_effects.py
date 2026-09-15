import numpy as np

from ruddy.factorial import analyze_mixed_effects


def test_random_intercept_mixed_model_recovers_fixed_effects(mixed_dataset):
    result = analyze_mixed_effects(
        mixed_dataset,
        group="batch",
        response="y",
        factors=("condition",),
        covariates=("x",),
        reml=False,
    )
    assert result.status.value == "ok"
    assert result.model_summary["converged"] is True
    assert result.model_summary["random_intercept_variance"] > 0.2
    assert 0 < result.model_summary["icc_random_intercept"] < 1
    assert len(result.fixed_effects) == 3


def test_mixed_model_requires_enough_groups(mixed_dataset):
    result = analyze_mixed_effects(
        mixed_dataset,
        group="batch",
        response="y",
        factors=("condition",),
        covariates=("x",),
        min_groups=30,
    )
    assert result.status.value == "degenerate"
    assert result.reason == "insufficient_groups"


def test_random_slope_must_be_numeric_covariate(mixed_dataset):
    try:
        analyze_mixed_effects(
            mixed_dataset,
            group="batch",
            response="y",
            factors=("condition",),
            covariates=("x",),
            random_slopes=("condition",),
        )
    except ValueError as exc:
        assert "random slopes" in str(exc).lower()
    else:
        msg = "Expected ValueError"
        raise AssertionError(msg)


def test_mixed_model_tracks_exclusions(mixed_dataset):
    from ruddy import TabularDataset

    frame = mixed_dataset.to_frame()
    frame.loc[0, "x"] = np.nan
    ds = TabularDataset(
        frame,
        id_column="id",
        role_overrides={"condition": "factor", "batch": "factor", "x": "covariate", "y": "response"},
    )
    result = analyze_mixed_effects(
        ds, group="batch", response="y", factors=("condition",), covariates=("x",), reml=False
    )
    assert len(result.exclusions) == 1


def test_numeric_random_slope_model_runs_and_estimates_slope_variance():
    import pandas as pd

    from ruddy import TabularDataset

    rng = np.random.default_rng(99)
    ng = 20
    m = 16
    n = ng * m
    batch = np.repeat([f"g{i}" for i in range(ng)], m)
    x = rng.normal(size=n)
    condition = np.tile(["A", "B"] * (m // 2), ng)
    ri = rng.normal(scale=1.0, size=ng)
    rs = rng.normal(loc=0.7, scale=0.45, size=ng)
    y = (
        1.4 * (np.asarray(condition) == "B")
        + np.repeat(ri, m)
        + np.repeat(rs, m) * x
        + rng.normal(scale=0.45, size=n)
    )
    frame = pd.DataFrame({"id": range(n), "batch": batch, "condition": condition, "x": x, "y": y})
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={"batch": "factor", "condition": "factor", "x": "covariate", "y": "response"},
    )
    result = analyze_mixed_effects(
        dataset,
        group="batch",
        response="y",
        factors=("condition",),
        covariates=("x",),
        random_slopes=("x",),
        reml=False,
    )
    assert result.status.value == "ok"
    slope_var = (
        result.variance_components.query(
            "component == 'random_effect_covariance' and row == 'X0' and column == 'X0'"
        )
        .iloc[0]
        .estimate
    )
    assert slope_var > 0.05
