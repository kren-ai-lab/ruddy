import numpy as np
import pandas as pd

from ruddy import TabularDataset, analyze_bayesian_eda


def _dataset(delta=2.0):
    rng = np.random.default_rng(10)
    n = 40
    frame = pd.DataFrame(
        {
            "id": [f"o{i}" for i in range(2 * n)],
            "y": np.r_[rng.normal(0, 0.5, n), rng.normal(delta, 0.5, n)],
            "g": ["A"] * n + ["B"] * n,
        }
    )
    return TabularDataset(frame, id_column="id", role_overrides={"y": "response", "g": "factor"})


def test_bayesian_mean_interval_contains_sample_region():
    r = analyze_bayesian_eda(_dataset(), draws=2000, random_state=1)
    row = r.means.iloc[0]
    assert row.credible_low < 1.0 < row.credible_high


def test_bayesian_group_difference_detects_direction():
    r = analyze_bayesian_eda(_dataset(), groups=("g",), draws=3000, random_state=2)
    row = r.mean_differences.iloc[0]
    assert row.posterior_mean < -1.5 and row.probability_direction > 0.99


def test_bayesian_deterministic_with_seed():
    a = analyze_bayesian_eda(_dataset(), groups=("g",), draws=500, random_state=4)
    b = analyze_bayesian_eda(_dataset(), groups=("g",), draws=500, random_state=4)
    pd.testing.assert_frame_equal(a.mean_differences, b.mean_differences)


def test_nonbinary_group_is_skipped_for_difference():
    ds = _dataset()
    f = ds.to_frame()
    f.loc[0, "g"] = "C"
    ds = TabularDataset(f, id_column="id", role_overrides={"y": "response", "g": "factor"})
    r = analyze_bayesian_eda(ds, groups=("g",), draws=500)
    assert r.mean_differences.loc[0, "status"] == "skipped"


def test_rope_probability_bounded():
    r = analyze_bayesian_eda(_dataset(delta=0.0), groups=("g",), draws=1000, rope=(-0.5, 0.5))
    p = float(r.mean_differences.loc[0, "rope_probability"])  # pyrefly: ignore[bad-argument-type]
    assert 0 <= p <= 1
