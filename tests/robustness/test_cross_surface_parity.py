from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pl_testing

from ruddy import (
    AnalysisConfig,
    FeatureMatrix,
    analyze,
    analyze_anomalies,
    analyze_pca,
    analyze_representation_similarity,
)
from ruddy.cli.commands.analyze import write_unified_result


def test_unified_pca_matches_standalone_exactly(robust_tabular):
    rng = np.random.default_rng(61)
    f = FeatureMatrix(
        rng.normal(size=(robust_tabular.n_observations, 8)),
        observation_ids=robust_tabular.observation_ids,
    )
    standalone = analyze_pca(f, n_components=4, scaling="standard", random_state=5)
    config = AnalysisConfig(
        enabled_blocks=("pca",),
        projection_n_components=4,
        projection_scaling="standard",
        random_state=5,
    )
    unified = analyze(robust_tabular, config=config, features=f).pca
    assert unified is not None
    pl_testing.assert_frame_equal(standalone.scores, unified.scores)
    pl_testing.assert_frame_equal(standalone.loadings, unified.loadings)
    pl_testing.assert_frame_equal(standalone.variance, unified.variance)


def test_unified_anomaly_matches_standalone_exactly(robust_tabular):
    rng = np.random.default_rng(62)
    f = FeatureMatrix(
        rng.normal(size=(robust_tabular.n_observations, 5)),
        observation_ids=robust_tabular.observation_ids,
    )
    standalone = analyze_anomalies(f, methods=("isolation_forest",), random_state=9)
    config = AnalysisConfig(
        enabled_blocks=("anomaly",),
        anomaly_methods=("isolation_forest",),
        random_state=9,
    )
    unified = analyze(robust_tabular, config=config, features=f).anomaly
    assert unified is not None
    pl_testing.assert_frame_equal(standalone.scores, unified.scores)
    pl_testing.assert_frame_equal(standalone.methods, unified.methods)


def test_unified_representation_matches_standalone_exactly(robust_tabular):
    rng = np.random.default_rng(63)
    latent = rng.normal(size=(robust_tabular.n_observations, 3))
    a = FeatureMatrix(latent @ rng.normal(size=(3, 7)), observation_ids=robust_tabular.observation_ids)
    b = FeatureMatrix(latent @ rng.normal(size=(3, 5)), observation_ids=robust_tabular.observation_ids)
    standalone = analyze_representation_similarity(
        a, b, cca_components=2, mantel_permutations=19, random_state=4
    )
    cfg = AnalysisConfig(
        enabled_blocks=("representation",),
        representation_cca_components=2,
        representation_mantel_permutations=19,
        random_state=4,
    )
    unified = analyze(robust_tabular, config=cfg, features=a, comparison_features=b).representation
    assert unified is not None
    pl_testing.assert_frame_equal(standalone.cka, unified.cka)
    pl_testing.assert_frame_equal(standalone.mantel, unified.mantel)
    assert unified.cca is not None
    pl_testing.assert_frame_equal(standalone.cca.correlations, unified.cca.correlations)


def test_cli_writer_preserves_unified_component_tables(tmp_path, robust_tabular):
    rng = np.random.default_rng(64)
    f = FeatureMatrix(
        rng.normal(size=(robust_tabular.n_observations, 6)),
        observation_ids=robust_tabular.observation_ids,
    )
    cfg = AnalysisConfig(
        enabled_blocks=("pca", "anomaly"),
        projection_n_components=3,
        anomaly_methods=("isolation_forest",),
        random_state=7,
    )
    result = analyze(robust_tabular, config=cfg, features=f)
    out = write_unified_result(result, tmp_path / "out")
    assert result.pca is not None
    pca_scores = pl.read_csv(out / "pca" / "pca_scores.csv", schema=result.pca.scores.schema)
    pl_testing.assert_frame_equal(
        pca_scores, result.pca.scores, check_exact=False, rel_tol=1e-12, abs_tol=1e-12
    )
    assert result.anomaly is not None
    assert result.anomaly.scores is not None
    anomaly_scores = pl.read_csv(out / "anomaly" / "anomaly_scores.csv", schema=result.anomaly.scores.schema)
    pl_testing.assert_frame_equal(
        anomaly_scores, result.anomaly.scores, check_exact=False, rel_tol=1e-12, abs_tol=1e-12
    )
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["executed_blocks"] == ["pca", "anomaly"]


def test_feature_alignment_never_uses_row_position(robust_tabular):
    ids = list(robust_tabular.observation_id_tuple)
    rng = np.random.default_rng(65)
    x = rng.normal(size=(len(ids), 4))
    order = np.arange(len(ids))[::-1]
    f = FeatureMatrix(x[order], observation_ids=[ids[i] for i in order])
    cfg = AnalysisConfig(enabled_blocks=("pca",), projection_n_components=2, feature_alignment="strict")
    result = analyze(robust_tabular, config=cfg, features=f)
    assert result.feature_alignment is not None
    assert result.feature_alignment.complete
    assert result.pca is not None
    assert result.pca.scores is not None
    components = ["PC1", "PC2"]
    scores_by_id = pl.DataFrame({"observation_id": ids}).join(
        result.pca.scores, on="observation_id", how="left"
    )
    # Project the original rows onto the returned axes to avoid PCA sign ambiguity.
    expected_scores = (x - x.mean(axis=0)) @ result.pca.loadings.select(components).to_numpy()
    np.testing.assert_allclose(
        scores_by_id.select(components).to_numpy(), expected_scores, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(scores_by_id["source_row_index"].to_numpy(), np.argsort(order))


def test_actual_cli_roundtrip_matches_standalone_univariate(tmp_path):
    from ruddy import TabularDataset, analyze_univariate
    from ruddy.cli.main import main

    rng = np.random.default_rng(66)
    frame = pd.DataFrame(
        {
            "id": [f"o{i}" for i in range(40)],
            "x": rng.normal(size=40),
            "y": rng.normal(size=40),
            "group": np.repeat(["A", "B"], 20),
        }
    )
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    out = tmp_path / "cli"
    assert (
        main(
            [
                "pipeline",
                str(source),
                "--id-column",
                "id",
                "--enable",
                "univariate",
                "--output-dir",
                str(out),
            ]
        )
        == 0
    )

    ds = TabularDataset(frame, id_column="id")
    standalone = analyze_univariate(ds)
    observed = pl.read_csv(
        out / "univariate" / "numeric_statistics.csv", schema=standalone.numeric_statistics.schema
    )
    pl_testing.assert_frame_equal(
        observed,
        standalone.numeric_statistics,
        check_exact=False,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )
