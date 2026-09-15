from pathlib import Path

import numpy as np
import pandas as pd

from ruddy.cli.main import main


def _write_group_data(path: Path, feature_path: Path) -> None:
    rng = np.random.default_rng(12)
    groups = np.repeat(["A", "B", "C"], 18)
    n = len(groups)
    x = rng.normal(size=n)
    y = (
        1.2 * (groups == "B")
        + 2.5 * (groups == "C")
        + 0.2 * x
        + rng.normal(scale=0.6, size=n)
    )
    pd.DataFrame({"id": range(n), "group": groups, "x": x, "y": y}).to_csv(
        path, index=False
    )
    feat = rng.normal(scale=0.4, size=(n, 3))
    feat[groups == "B", 0] += 2
    feat[groups == "C", 0] += 4
    pd.DataFrame(
        {"id": range(n), "f1": feat[:, 0], "f2": feat[:, 1], "f3": feat[:, 2]}
    ).to_csv(feature_path, index=False)


def _write_mixed_data(path: Path) -> None:
    rng = np.random.default_rng(8)
    ng = 14
    m = 12
    n = ng * m
    batch = np.repeat([f"b{i}" for i in range(ng)], m)
    condition = np.tile(["A", "B"] * (m // 2), ng)
    x = rng.normal(size=n)
    u = rng.normal(scale=1.2, size=ng)
    y = (
        1.5 * (np.asarray(condition) == "B")
        + 0.5 * x
        + np.repeat(u, m)
        + rng.normal(scale=0.5, size=n)
    )
    pd.DataFrame(
        {"id": range(n), "batch": batch, "condition": condition, "x": x, "y": y}
    ).to_csv(path, index=False)


def test_permanova_cli(tmp_path):
    data = tmp_path / "data.csv"
    feat = tmp_path / "feat.csv"
    out = tmp_path / "perm"
    _write_group_data(data, feat)
    code = main(
        [
            "model",
            "permanova",
            str(data),
            "--id-column",
            "id",
            "--permanova-factor",
            "group",
            "--feature-input",
            str(feat),
            "--feature-id-column",
            "id",
            "--feature-column",
            "f1",
            "--feature-column",
            "f2",
            "--feature-column",
            "f3",
            "--permanova-permutations",
            "19",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "permutation_group_summary.csv").exists()
    assert (out / "permdisp_distances.csv").exists()


def test_posthoc_cli(tmp_path):
    data = tmp_path / "data.csv"
    feat = tmp_path / "feat.csv"
    out = tmp_path / "post"
    _write_group_data(data, feat)
    code = main(
        [
            "analyze",
            "posthoc",
            str(data),
            "--id-column",
            "id",
            "--posthoc-response",
            "y",
            "--posthoc-factor",
            "group",
            "--posthoc-method",
            "games_howell",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    table = pd.read_csv(out / "posthoc_comparisons.csv")
    assert set(table.method) == {"games_howell"}


def test_marginal_means_cli(tmp_path):
    data = tmp_path / "data.csv"
    feat = tmp_path / "feat.csv"
    out = tmp_path / "emm"
    _write_group_data(data, feat)
    code = main(
        [
            "model",
            "marginal-means",
            str(data),
            "--id-column",
            "id",
            "--marginal-response",
            "y",
            "--marginal-factor",
            "group",
            "--marginal-covariate",
            "x",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "marginal_means.csv").exists()
    assert (out / "marginal_contrasts.csv").exists()


def test_mixed_effects_cli(tmp_path):
    data = tmp_path / "mixed.csv"
    out = tmp_path / "mixed"
    _write_mixed_data(data)
    code = main(
        [
            "model",
            "mixed-effects",
            str(data),
            "--id-column",
            "id",
            "--mixed-group",
            "batch",
            "--mixed-response",
            "y",
            "--mixed-factor",
            "condition",
            "--mixed-covariate",
            "x",
            "--mixed-ml",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "mixed_fixed_effects.csv").exists()
    assert (out / "mixed_model_summary.json").exists()


def test_unified_cli_phase9b_blocks(tmp_path):
    data = tmp_path / "data.csv"
    feat = tmp_path / "feat.csv"
    out = tmp_path / "all"
    _write_group_data(data, feat)
    code = main(
        [
            "analyze",
            "run",
            str(data),
            "--id-column",
            "id",
            "--response",
            "y",
            "--group",
            "group",
            "--enable",
            "permanova",
            "--enable",
            "posthoc",
            "--enable",
            "marginal_means",
            "--feature-input",
            str(feat),
            "--feature-id-column",
            "id",
            "--feature-column",
            "f1",
            "--feature-column",
            "f2",
            "--feature-column",
            "f3",
            "--permanova-factor",
            "group",
            "--permanova-permutations",
            "9",
            "--posthoc-response",
            "y",
            "--posthoc-factor",
            "group",
            "--marginal-response",
            "y",
            "--marginal-factor",
            "group",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    for name in ("permanova", "posthoc", "marginal_means"):
        assert (out / name).is_dir()
