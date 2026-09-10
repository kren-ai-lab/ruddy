from __future__ import annotations

import json
import numpy as np
import pandas as pd

from ruddy import AnalysisConfig, FeatureMatrix, analyze, analyze_anomalies, analyze_pca, analyze_representation_similarity
from ruddy.cli.commands.analyze import write_unified_result


def test_unified_pca_matches_standalone_exactly(robust_tabular):
    rng = np.random.default_rng(61)
    f = FeatureMatrix(rng.normal(size=(robust_tabular.n_observations, 8)), observation_ids=robust_tabular.observation_ids)
    standalone = analyze_pca(f, n_components=4, scaling="standard", random_state=5)
    config = AnalysisConfig(enabled_blocks=("pca",), projection_n_components=4, projection_scaling="standard", random_state=5)
    unified = analyze(robust_tabular, config=config, features=f).pca
    pd.testing.assert_frame_equal(standalone.scores, unified.scores)
    pd.testing.assert_frame_equal(standalone.loadings, unified.loadings)
    pd.testing.assert_frame_equal(standalone.variance, unified.variance)


def test_unified_anomaly_matches_standalone_exactly(robust_tabular):
    rng = np.random.default_rng(62)
    f = FeatureMatrix(rng.normal(size=(robust_tabular.n_observations, 5)), observation_ids=robust_tabular.observation_ids)
    standalone = analyze_anomalies(f, methods=("isolation_forest",), random_state=9)
    config = AnalysisConfig(enabled_blocks=("anomaly",), anomaly_methods=("isolation_forest",), random_state=9)
    unified = analyze(robust_tabular, config=config, features=f).anomaly
    pd.testing.assert_frame_equal(standalone.scores, unified.scores)
    pd.testing.assert_frame_equal(standalone.methods, unified.methods)


def test_unified_representation_matches_standalone_exactly(robust_tabular):
    rng = np.random.default_rng(63)
    latent = rng.normal(size=(robust_tabular.n_observations, 3))
    a = FeatureMatrix(latent @ rng.normal(size=(3, 7)), observation_ids=robust_tabular.observation_ids)
    b = FeatureMatrix(latent @ rng.normal(size=(3, 5)), observation_ids=robust_tabular.observation_ids)
    standalone = analyze_representation_similarity(a, b, cca_components=2, mantel_permutations=19, random_state=4)
    cfg = AnalysisConfig(enabled_blocks=("representation",), representation_cca_components=2, representation_mantel_permutations=19, random_state=4)
    unified = analyze(robust_tabular, config=cfg, features=a, comparison_features=b).representation
    pd.testing.assert_frame_equal(standalone.cka, unified.cka)
    pd.testing.assert_frame_equal(standalone.mantel, unified.mantel)
    pd.testing.assert_frame_equal(standalone.cca.correlations, unified.cca.correlations)


def test_cli_writer_preserves_unified_component_tables(tmp_path, robust_tabular):
    rng = np.random.default_rng(64)
    f = FeatureMatrix(rng.normal(size=(robust_tabular.n_observations, 6)), observation_ids=robust_tabular.observation_ids)
    cfg = AnalysisConfig(enabled_blocks=("pca", "anomaly"), projection_n_components=3, anomaly_methods=("isolation_forest",), random_state=7)
    result = analyze(robust_tabular, config=cfg, features=f)
    out = write_unified_result(result, tmp_path / "out")
    pca_scores = pd.read_csv(out / "pca" / "pca_scores.csv")
    anomaly_scores = pd.read_csv(out / "anomaly" / "anomaly_scores.csv")
    pd.testing.assert_frame_equal(pca_scores, result.pca.scores, check_dtype=False, rtol=1e-12, atol=1e-12)
    expected_anomaly = result.anomaly.scores.copy()
    for column in expected_anomaly.select_dtypes(include="object").columns:
        expected_anomaly[column] = expected_anomaly[column].fillna("")
        anomaly_scores[column] = anomaly_scores[column].fillna("")
    pd.testing.assert_frame_equal(anomaly_scores, expected_anomaly, check_dtype=False, rtol=1e-12, atol=1e-12)
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["executed_blocks"] == ["pca", "anomaly"]


def test_feature_alignment_never_uses_row_position(robust_tabular):
    ids = robust_tabular.observation_ids.to_list()
    rng = np.random.default_rng(65)
    x = rng.normal(size=(len(ids), 4))
    order = np.arange(len(ids))[::-1]
    f = FeatureMatrix(x[order], observation_ids=[ids[i] for i in order])
    cfg = AnalysisConfig(enabled_blocks=("pca",), projection_n_components=2, feature_alignment="strict")
    result = analyze(robust_tabular, config=cfg, features=f)
    assert result.feature_alignment.complete
    assert set(result.pca.scores.observation_id) == set(ids)


def test_actual_cli_roundtrip_matches_standalone_univariate(tmp_path):
    from ruddy import TabularDataset, analyze_univariate
    from ruddy.cli.main import main

    rng = np.random.default_rng(66)
    frame = pd.DataFrame({
        "id": [f"o{i}" for i in range(40)],
        "x": rng.normal(size=40),
        "y": rng.normal(size=40),
        "group": np.repeat(["A", "B"], 20),
    })
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    out = tmp_path / "cli"
    assert main([
        "analyze", str(source), "--id-column", "id",
        "--enable", "univariate", "--output-dir", str(out),
    ]) == 0

    ds = TabularDataset(frame, id_column="id")
    standalone = analyze_univariate(ds)
    observed = pd.read_csv(out / "univariate" / "numeric_statistics.csv")
    expected = standalone.numeric_statistics.copy()
    for column in expected.select_dtypes(include="object").columns:
        expected[column] = expected[column].fillna("")
        observed[column] = observed[column].fillna("")
    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
