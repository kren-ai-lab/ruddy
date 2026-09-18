import numpy as np
import pandas as pd

from ruddy import FeatureMatrix, TabularDataset
from ruddy.cli.main import main


def _bundle():
    rng = np.random.default_rng(22)
    n = 36
    ids = [f"o{i}" for i in range(n)]
    frame = pd.DataFrame({"id": ids, "y": rng.normal(size=n), "g": ["A"] * (n // 2) + ["B"] * (n // 2)})
    ds = TabularDataset(frame, id_column="id", role_overrides={"y": "response", "g": "factor"})
    x = rng.normal(size=(n, 3))
    y = x @ rng.normal(size=(3, 4)) + 0.1 * rng.normal(size=(n, 4))
    return (
        ds,
        FeatureMatrix(x, observation_ids=ids, feature_names=["x1", "x2", "x3"]),
        FeatureMatrix(y, observation_ids=ids, feature_names=["y1", "y2", "y3", "y4"]),
    )


def test_representation_cli_writes_artifacts(tmp_path):
    _ds, x, y = _bundle()
    xa = tmp_path / "x.csv"
    ya = tmp_path / "y.csv"
    out = tmp_path / "out"
    pd.DataFrame(
        {
            "id": x.observation_ids,
            **{n: x.to_array()[:, i] for i, n in enumerate(x.feature_names)},
        }
    ).to_csv(xa, index=False)
    pd.DataFrame(
        {
            "id": y.observation_ids,
            **{n: y.to_array()[:, i] for i, n in enumerate(y.feature_names)},
        }
    ).to_csv(ya, index=False)
    code = main(
        [
            "analyze",
            "representation",
            str(xa),
            str(ya),
            "--x-id-column",
            "id",
            "--y-id-column",
            "id",
            "--cca-components",
            "2",
            "--mantel-permutations",
            "9",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0 and (out / "cka.csv").exists() and (out / "cca_correlations.csv").exists()
    assert (out / "cca_x_weights.csv").exists()
    assert (out / "cca_x_scores.csv").exists()
    assert not (out / "representation_exclusions.csv").exists()


def test_representation_cli_writes_exclusions_when_nonfinite(tmp_path):
    _ds, x, y = _bundle()
    xa = tmp_path / "x.csv"
    ya = tmp_path / "y.csv"
    out = tmp_path / "out_excl"
    x_arr = x.to_array()
    x_arr[0, 0] = np.nan
    pd.DataFrame(
        {
            "id": x.observation_ids,
            **{n: x_arr[:, i] for i, n in enumerate(x.feature_names)},
        }
    ).to_csv(xa, index=False)
    pd.DataFrame(
        {
            "id": y.observation_ids,
            **{n: y.to_array()[:, i] for i, n in enumerate(y.feature_names)},
        }
    ).to_csv(ya, index=False)
    code = main(
        [
            "analyze",
            "representation",
            str(xa),
            str(ya),
            "--x-id-column",
            "id",
            "--y-id-column",
            "id",
            "--cca-components",
            "2",
            "--mantel-permutations",
            "0",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0 and (out / "representation_exclusions.csv").exists()


def test_representation_cli_stdout(tmp_path, capsys):
    _ds, x, y = _bundle()
    xa = tmp_path / "x.csv"
    ya = tmp_path / "y.csv"
    pd.DataFrame(
        {
            "id": x.observation_ids,
            **{n: x.to_array()[:, i] for i, n in enumerate(x.feature_names)},
        }
    ).to_csv(xa, index=False)
    pd.DataFrame(
        {
            "id": y.observation_ids,
            **{n: y.to_array()[:, i] for i, n in enumerate(y.feature_names)},
        }
    ).to_csv(ya, index=False)
    code = main(
        [
            "analyze",
            "representation",
            str(xa),
            str(ya),
            "--x-id-column",
            "id",
            "--y-id-column",
            "id",
            "--cca-components",
            "2",
            "--mantel-permutations",
            "9",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "cka" in captured.out
    assert "mantel" in captured.out
    assert "cca_status" in captured.out


def test_compositional_cli_writes_artifacts(tmp_path):
    import polars as pl

    rng = np.random.default_rng(3)
    p = tmp_path / "c.csv"
    out = tmp_path / "c_out"
    frame = pd.DataFrame(np.abs(rng.normal(size=(20, 3))) + 0.2, columns=pd.Index(["a", "b", "c"]))
    frame.insert(0, "id", [f"o{i}" for i in range(20)])
    frame.to_csv(p, index=False)
    code = main(
        [
            "analyze",
            "compositional",
            str(p),
            "--id-column",
            "id",
            "--compositional-transform",
            "clr",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0 and (out / "compositional_transformed.csv").exists()
    assert (out / "variation_matrix.csv").exists()
    assert (out / "aitchison_distances.csv").exists()
    assert (out / "zero_replacement.csv").exists()

    var_df = pl.read_csv(out / "variation_matrix.csv")
    assert var_df.columns == ["feature", "a", "b", "c"]
    dist_df = pl.read_csv(out / "aitchison_distances.csv")
    assert dist_df.columns == ["observation_id", *[f"o{i}" for i in range(20)]]
    trans_df = pl.read_csv(out / "compositional_transformed.csv")
    assert trans_df.columns == ["observation_id", "clr_a", "clr_b", "clr_c"]


def test_compositional_cli_stdout(tmp_path, capsys):
    rng = np.random.default_rng(3)
    p = tmp_path / "c.csv"
    frame = pd.DataFrame(np.abs(rng.normal(size=(20, 3))) + 0.2, columns=pd.Index(["a", "b", "c"]))
    frame.insert(0, "id", [f"o{i}" for i in range(20)])
    frame.to_csv(p, index=False)
    code = main(
        [
            "analyze",
            "compositional",
            str(p),
            "--id-column",
            "id",
            "--compositional-transform",
            "clr",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "transform" in captured.out
    assert "shape" in captured.out


def test_bayesian_cli_writes_artifacts(tmp_path):
    ds, _, _ = _bundle()
    p = tmp_path / "d.csv"
    out = tmp_path / "b"
    ds.to_frame().to_csv(p, index=False)
    code = main(
        [
            "analyze",
            "bayesian",
            str(p),
            "--id-column",
            "id",
            "--response",
            "y",
            "--factor",
            "g",
            "--bayesian-variable",
            "y",
            "--bayesian-group",
            "g",
            "--bayesian-draws",
            "500",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0 and (out / "bayesian_means.csv").exists()


def test_anomaly_cli_writes_artifacts(tmp_path):
    _, x, _ = _bundle()
    p = tmp_path / "x.csv"
    out = tmp_path / "a"
    pd.DataFrame(
        {
            "id": x.observation_ids,
            **{n: x.to_array()[:, i] for i, n in enumerate(x.feature_names)},
        }
    ).to_csv(p, index=False)
    code = main(
        [
            "analyze",
            "anomaly",
            str(p),
            "--id-column",
            "id",
            "--anomaly-method",
            "isolation_forest",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0 and (out / "anomaly_scores.csv").exists()
