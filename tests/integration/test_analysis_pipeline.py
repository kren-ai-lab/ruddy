from __future__ import annotations

import numpy as np
import polars.testing as pl_testing
import pytest
from pandas.testing import assert_frame_equal

from ruddy import (
    AnalysisBlock,
    AnalysisConfig,
    FeatureMatrix,
    analyze,
    analyze_bivariate,
    analyze_factorial,
    analyze_grouped_responses,
    analyze_manova,
    analyze_multivariate,
    analyze_outliers,
    analyze_univariate,
    profile_dataset,
)
from ruddy.core import AlignmentError


def test_default_pipeline_runs_descriptive_blocks_only(pipeline_dataset):
    result = analyze(pipeline_dataset)
    assert result.executed_blocks == (AnalysisBlock.PROFILING, AnalysisBlock.UNIVARIATE)
    assert result.profiling is not None
    assert result.univariate is not None
    assert result.bivariate is None
    assert result.outliers is None
    assert result.factorial is None
    assert result.dependence is None
    assert result.intervals is None


def test_single_block_does_not_trigger_unrelated_components(pipeline_dataset):
    config = AnalysisConfig(enabled_blocks=("bivariate",))
    result = analyze(pipeline_dataset, config=config)
    assert result.executed_blocks == (AnalysisBlock.BIVARIATE,)
    assert result.bivariate is not None
    assert result.profiling is None
    assert result.univariate is None
    assert result.outliers is None
    assert result.groups is None


def test_profiling_matches_standalone(pipeline_dataset):
    config = AnalysisConfig(enabled_blocks=("profiling",))
    result = analyze(pipeline_dataset, config=config)
    standalone = profile_dataset(pipeline_dataset, **config.profiling_kwargs())
    assert result.profiling is not None
    assert result.profiling.overview == standalone.overview
    pl_testing.assert_frame_equal(result.profiling.columns, standalone.columns)
    pl_testing.assert_frame_equal(result.profiling.missingness, standalone.missingness)


def test_univariate_matches_standalone(pipeline_dataset):
    config = AnalysisConfig(enabled_blocks=("univariate",))
    result = analyze(pipeline_dataset, config=config)
    standalone = analyze_univariate(pipeline_dataset, **config.univariate_kwargs())
    assert result.univariate is not None
    pl_testing.assert_frame_equal(result.univariate.numeric_statistics, standalone.numeric_statistics)
    pl_testing.assert_frame_equal(result.univariate.categorical_statistics, standalone.categorical_statistics)


def test_bivariate_matches_standalone(pipeline_dataset):
    config = AnalysisConfig(enabled_blocks=("bivariate",), correlations=("pearson", "spearman"))
    result = analyze(pipeline_dataset, config=config)
    standalone = analyze_bivariate(pipeline_dataset, **config.bivariate_kwargs())
    assert result.bivariate is not None
    assert_frame_equal(result.bivariate.correlations, standalone.correlations)
    assert_frame_equal(result.bivariate.comparisons, standalone.comparisons)
    assert_frame_equal(result.bivariate.categorical_associations, standalone.categorical_associations)


def test_groups_matches_standalone(pipeline_dataset):
    config = AnalysisConfig(
        enabled_blocks=("groups",),
        responses=("y1",),
        groups=("factor",),
    )
    result = analyze(pipeline_dataset, config=config)
    standalone = analyze_grouped_responses(pipeline_dataset, **config.grouped_kwargs())
    assert result.groups is not None
    assert_frame_equal(result.groups.numeric_summaries, standalone.numeric_summaries)
    assert_frame_equal(result.groups.numeric_comparisons, standalone.numeric_comparisons)


def test_outliers_matches_standalone(pipeline_dataset):
    config = AnalysisConfig(enabled_blocks=("outliers",), include_outlier_flags=True)
    result = analyze(pipeline_dataset, config=config)
    standalone = analyze_outliers(pipeline_dataset, **config.outlier_kwargs())
    assert result.outliers is not None
    assert_frame_equal(result.outliers.summaries, standalone.summaries)
    assert_frame_equal(result.outliers.flags, standalone.flags)
    assert_frame_equal(result.outliers.quality, standalone.quality)


def test_feature_block_requires_feature_matrix(pipeline_dataset):
    config = AnalysisConfig(enabled_blocks=("pca",))
    with pytest.raises(ValueError, match="FeatureMatrix is required"):
        analyze(pipeline_dataset, config=config)


def test_strict_feature_alignment_rejects_mismatch(pipeline_dataset, pipeline_features):
    bad = FeatureMatrix(
        pipeline_features.to_array()[:-1],
        observation_ids=pipeline_features.observation_ids[:-1],
        feature_names=pipeline_features.feature_names,
    )
    config = AnalysisConfig(enabled_blocks=("pca",), feature_alignment="strict")
    with pytest.raises(AlignmentError):
        analyze(pipeline_dataset, config=config, features=bad)


def test_partial_feature_alignment_is_observable(pipeline_dataset, pipeline_features):
    partial = FeatureMatrix(
        pipeline_features.to_array()[:-3],
        observation_ids=pipeline_features.observation_ids[:-3],
        feature_names=pipeline_features.feature_names,
    )
    config = AnalysisConfig(enabled_blocks=("pca",), feature_alignment="partial")
    result = analyze(pipeline_dataset, config=config, features=partial)
    assert result.feature_alignment is not None
    assert result.feature_alignment.covered_count == pipeline_dataset.n_observations - 3
    assert len(result.feature_alignment.missing_ids) == 3


def test_multivariate_matches_standalone(pipeline_dataset, pipeline_features):
    config = AnalysisConfig(enabled_blocks=("multivariate",), mahalanobis_include_robust=False)
    result = analyze(pipeline_dataset, config=config, features=pipeline_features)
    standalone = analyze_multivariate(pipeline_features, **config.multivariate_kwargs())
    assert result.multivariate is not None
    assert result.multivariate.covariance is not None
    assert_frame_equal(result.multivariate.covariance.covariance, standalone.covariance.covariance)
    assert result.multivariate.collinearity is not None
    assert_frame_equal(result.multivariate.collinearity.features, standalone.collinearity.features)
    assert result.multivariate.mahalanobis is not None
    assert_frame_equal(result.multivariate.mahalanobis.distances, standalone.mahalanobis.distances)


def test_manova_matches_standalone(pipeline_dataset):
    config = AnalysisConfig(
        enabled_blocks=("manova",),
        responses=("y1", "y2"),
        groups=("factor",),
    )
    result = analyze(pipeline_dataset, config=config)
    standalone = analyze_manova(
        pipeline_dataset,
        responses=("y1", "y2"),
        factors=("factor",),
        covariates=("x",),
        **config.manova_kwargs(),
    )
    assert result.manova is not None
    assert_frame_equal(result.manova.tests, standalone.tests)
    assert_frame_equal(result.manova.factor_levels, standalone.factor_levels)


def test_factorial_matches_standalone_with_formula(pipeline_dataset):
    config = AnalysisConfig(
        enabled_blocks=("factorial",),
        factorial_formula="y1 ~ factor * batch + x",
        factorial_ss_type=3,
    )
    result = analyze(pipeline_dataset, config=config)
    standalone = analyze_factorial(
        pipeline_dataset,
        formula="y1 ~ factor * batch + x",
        **config.factorial_kwargs(),
    )
    assert result.factorial is not None
    assert_frame_equal(result.factorial.effects, standalone.effects)
    assert_frame_equal(result.factorial.coefficients, standalone.coefficients)
    assert result.factorial.design is not None
    assert result.factorial.design.resolved_formula == standalone.design.resolved_formula


def test_factorial_explicit_model_uses_configured_interaction(pipeline_dataset):
    config = AnalysisConfig(
        enabled_blocks=("factorial",),
        responses=("y1",),
        groups=("factor", "batch"),
        factorial_interactions=(("factor", "batch"),),
    )
    result = analyze(pipeline_dataset, config=config)
    assert result.factorial is not None
    terms = set(result.factorial.effects["term"].astype(str))
    assert any("factor" in term and "batch" in term for term in terms)


def test_manova_requires_explicit_multivariate_response_selection(pipeline_dataset):
    config = AnalysisConfig(enabled_blocks=("manova",), responses=("y1",), groups=("factor",))
    with pytest.raises(ValueError, match="at least two"):
        analyze(pipeline_dataset, config=config)


def test_factorial_requires_explicit_response_semantics(pipeline_dataset):
    config = AnalysisConfig(enabled_blocks=("factorial",))
    with pytest.raises(ValueError, match="exactly one configured response"):
        analyze(pipeline_dataset, config=config)


def test_component_mapping_contains_only_executed_blocks(pipeline_dataset):
    result = analyze(pipeline_dataset, config=AnalysisConfig(enabled_blocks=("profiling", "outliers")))
    assert tuple(result.components) == (AnalysisBlock.PROFILING, AnalysisBlock.OUTLIERS)
    with pytest.raises(TypeError):
        result.components[AnalysisBlock.BIVARIATE] = None  # pyrefly: ignore[unsupported-operation]


def test_component_accessor_accepts_string(pipeline_dataset):
    result = analyze(pipeline_dataset, config=AnalysisConfig(enabled_blocks=("profiling",)))
    assert result.component("profiling") is result.profiling
    assert result.component("bivariate") is None


def test_summary_is_serialization_safe(pipeline_dataset):
    result = analyze(pipeline_dataset, config=AnalysisConfig(enabled_blocks=("profiling",)))
    summary = result.summary()
    assert summary["executed_blocks"] == ["profiling"]
    assert summary["components"] == {"profiling": True}
    assert summary["provenance"]["analysis"] == "unified"


def test_orchestration_does_not_mutate_sources(pipeline_frame, pipeline_features):
    original_frame = pipeline_frame.copy(deep=True)
    original_matrix = pipeline_features.to_array()
    dataset = __import__("ruddy").TabularDataset(
        pipeline_frame,
        id_column="id",
        role_overrides={"y1": "response", "factor": "factor", "sequence": "excluded"},
    )
    analyze(
        dataset,
        config=AnalysisConfig(enabled_blocks=("profiling", "pca"), projection_n_components=2),
        features=pipeline_features,
    )
    assert_frame_equal(pipeline_frame, original_frame)
    np.testing.assert_array_equal(pipeline_features.to_array(), original_matrix)


def test_same_seed_produces_identical_pca_scientific_outputs(pipeline_dataset, pipeline_features):
    config = AnalysisConfig(enabled_blocks=("pca",), projection_n_components=3, random_state=17)
    first = analyze(pipeline_dataset, config=config, features=pipeline_features)
    second = analyze(pipeline_dataset, config=config, features=pipeline_features)
    assert second.pca is not None
    assert first.pca is not None
    assert_frame_equal(first.pca.scores, second.pca.scores)
    assert_frame_equal(first.pca.variance, second.pca.variance)


def test_multiple_blocks_preserve_requested_order(pipeline_dataset, pipeline_features):
    config = AnalysisConfig(enabled_blocks=("outliers", "profiling", "pca"), projection_n_components=2)
    result = analyze(pipeline_dataset, config=config, features=pipeline_features)
    assert result.executed_blocks == (
        AnalysisBlock.OUTLIERS,
        AnalysisBlock.PROFILING,
        AnalysisBlock.PCA,
    )
