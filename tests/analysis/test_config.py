from __future__ import annotations

import pytest

from ruddy import (
    AlignmentMode,
    AnalysisBlock,
    AnalysisConfig,
    ColumnKind,
    ColumnRole,
    OutlierMethod,
    PAdjustMethod,
    ScalingMethod,
)


def test_analysis_config_normalizes_and_freezes_overrides() -> None:
    config = AnalysisConfig(
        id_column="id",
        role_overrides={"group": "factor"},
        kind_overrides={"code": "categorical"},
        annotation_alignment="partial",
        random_state=11,
    )

    assert config.role_overrides["group"] is ColumnRole.FACTOR
    assert config.kind_overrides["code"] is ColumnKind.CATEGORICAL
    assert config.annotation_alignment is AlignmentMode.PARTIAL
    assert config.dataset_kwargs()["id_column"] == "id"

    with pytest.raises(TypeError):
        config.role_overrides["x"] = ColumnRole.RESPONSE  # pyrefly: ignore[unsupported-operation]


def test_descriptive_controls_are_validated() -> None:
    config = AnalysisConfig(
        quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        min_numeric_n=4,
        max_category_levels=10,
        max_missingness_patterns=5,
        max_pairwise_columns=50,
    )
    assert config.univariate_kwargs()["min_numeric_n"] == 4
    assert config.profiling_kwargs()["max_pairwise_columns"] == 50

    with pytest.raises(ValueError, match=r"include 0\.25"):
        AnalysisConfig(quantiles=(0.1, 0.5, 0.9))


def test_response_and_group_selection_is_separate_from_roles() -> None:
    config = AnalysisConfig(
        responses=("response_a", "response_b"),
        groups=("group_a",),
    )
    assert config.responses == ("response_a", "response_b")
    assert config.groups == ("group_a",)
    assert config.grouped_kwargs()["responses"] == ("response_a", "response_b")
    assert config.grouped_kwargs()["groups"] == ("group_a",)

    with pytest.raises(ValueError, match="both response and group"):
        AnalysisConfig(responses=("x",), groups=("x",))


def test_outlier_controls_are_normalized() -> None:
    config = AnalysisConfig(
        outlier_methods=("robust_z", "iqr"),
        outlier_iqr_multiplier=2.0,
        outlier_robust_z_threshold=4.0,
        include_outlier_flags=True,
    )
    assert config.outlier_methods == (OutlierMethod.ROBUST_Z, OutlierMethod.IQR)
    assert config.outlier_kwargs() == {
        "methods": (OutlierMethod.ROBUST_Z, OutlierMethod.IQR),
        "min_numeric_n": 3,
        "iqr_multiplier": 2.0,
        "robust_z_threshold": 4.0,
        "include_flags": True,
    }


def test_invalid_outlier_controls_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        AnalysisConfig(outlier_methods=())
    with pytest.raises(ValueError, match="duplicates"):
        AnalysisConfig(outlier_methods=("iqr", "iqr"))
    with pytest.raises(ValueError, match="greater than zero"):
        AnalysisConfig(outlier_iqr_multiplier=0)
    with pytest.raises(ValueError, match="greater than zero"):
        AnalysisConfig(outlier_robust_z_threshold=-1)


def test_projection_defaults_do_not_scale():
    config = AnalysisConfig()
    assert config.projection_scaling is ScalingMethod.NONE
    assert config.pca_kwargs()["scaling"] is ScalingMethod.NONE
    assert config.projection_n_components == 2


def test_projection_kwargs():
    config = AnalysisConfig(
        random_state=4,
        projection_scaling="robust",
        projection_n_components=3,
        projection_metric="cosine",
        tsne_perplexity=8,
        tsne_max_iter=500,
        umap_n_neighbors=7,
        umap_min_dist=0.2,
    )
    assert config.tsne_kwargs()["metric"] == "cosine"
    assert config.tsne_kwargs()["perplexity"] == 8
    assert config.umap_kwargs()["n_neighbors"] == 7
    assert config.umap_kwargs()["min_dist"] == 0.2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"projection_n_components": 0},
        {"tsne_perplexity": 0},
        {"tsne_max_iter": 200},
        {"umap_n_neighbors": 1},
        {"umap_min_dist": 1.1},
    ],
)
def test_invalid_projection_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AnalysisConfig(**kwargs)


def test_multivariate_config_kwargs_are_explicit():
    config = AnalysisConfig(
        multivariate_scaling="standard",
        multivariate_include_spearman=False,
        mahalanobis_threshold_quantile=0.99,
        mahalanobis_include_robust=False,
        multivariate_max_covariance_features=80,
        multivariate_max_collinearity_features=40,
        multivariate_max_mahalanobis_features=30,
        random_state=17,
    )
    kwargs = config.multivariate_kwargs()
    assert kwargs["scaling"] is ScalingMethod.STANDARD
    assert kwargs["include_spearman"] is False
    assert kwargs["mahalanobis_threshold_quantile"] == 0.99
    assert kwargs["include_robust_mahalanobis"] is False
    assert kwargs["random_state"] == 17


def test_manova_config_guards():
    config = AnalysisConfig(manova_max_responses=12, manova_max_factor_levels=8, manova_min_level_n=4)
    assert config.manova_kwargs() == {"max_responses": 12, "max_factor_levels": 8, "min_level_n": 4}
    with pytest.raises(ValueError):
        AnalysisConfig(mahalanobis_threshold_quantile=1.0)


def test_factorial_config_kwargs_are_explicit():
    config = AnalysisConfig(
        factorial_ss_type=3,
        factorial_p_adjust="fdr_bh",
        factorial_robust_covariance="HC3",
        factorial_min_cell_n=4,
        factorial_max_factor_levels=12,
        factorial_max_design_cells=1000,
        factorial_max_design_columns=250,
        factorial_max_interaction_order=3,
        factorial_diagnostic_alpha=0.01,
        factorial_condition_number_threshold=50.0,
    )
    kwargs = config.factorial_kwargs()
    assert kwargs["ss_type"] == 3
    assert kwargs["p_adjust"] is PAdjustMethod.FDR_BH
    assert kwargs["robust_covariance"] == "hc3"
    assert kwargs["min_cell_n"] == 4
    assert kwargs["diagnostic_alpha"] == 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"factorial_ss_type": 1},
        {"factorial_robust_covariance": "sandwich"},
        {"factorial_min_cell_n": 0},
        {"factorial_max_factor_levels": 1},
        {"factorial_max_design_cells": 0},
        {"factorial_max_design_columns": 1},
        {"factorial_max_interaction_order": 1},
        {"factorial_diagnostic_alpha": 1.0},
        {"factorial_condition_number_threshold": 0.0},
    ],
)
def test_factorial_config_guards(kwargs):
    with pytest.raises(ValueError):
        AnalysisConfig(**kwargs)


def test_default_enabled_blocks_are_descriptive_only():
    config = AnalysisConfig()
    assert config.enabled_blocks == (AnalysisBlock.PROFILING, AnalysisBlock.UNIVARIATE)


def test_enabled_blocks_normalize_strings():
    config = AnalysisConfig(enabled_blocks=("profiling", "bivariate", "pca"))
    assert config.enabled_blocks == (
        AnalysisBlock.PROFILING,
        AnalysisBlock.BIVARIATE,
        AnalysisBlock.PCA,
    )


def test_enabled_blocks_cannot_be_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        AnalysisConfig(enabled_blocks=())


def test_enabled_blocks_cannot_repeat():
    with pytest.raises(ValueError, match="duplicates"):
        AnalysisConfig(enabled_blocks=("profiling", "profiling"))


def test_feature_alignment_normalizes():
    assert AnalysisConfig(feature_alignment="partial").feature_alignment is AlignmentMode.PARTIAL


def test_factorial_interactions_are_immutable_tuples():
    interactions = [["A", "B"]]
    # Mutable inputs are accepted and normalized at runtime.
    config = AnalysisConfig(factorial_interactions=interactions)  # pyrefly: ignore[bad-argument-type]
    assert config.factorial_interactions == (("A", "B"),)

    interactions[0].append("C")
    interactions.append(["B", "C"])
    assert config.factorial_interactions == (("A", "B"),)


def test_factorial_interaction_requires_two_predictors():
    with pytest.raises(ValueError, match="at least two"):
        AnalysisConfig(factorial_interactions=(("A",),))


def test_interval_and_dependence_controls_are_validated() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        AnalysisConfig(confidence_level=1.2)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        AnalysisConfig(bootstrap_resamples=20)
    with pytest.raises(ValueError, match="dependence_n_permutations"):
        AnalysisConfig(dependence_n_permutations=-1)
    with pytest.raises(ValueError, match="duplicates"):
        AnalysisConfig(dependence_partial_covariates=("z", "z"))


def test_group_model_blocks_and_methods_are_normalized():
    config = AnalysisConfig(
        enabled_blocks=("permanova", "posthoc", "marginal_means", "mixed_effects"),
        posthoc_methods=("games_howell",),
        marginal_p_adjust="none",
    )
    assert list(config.enabled_blocks) == [
        "permanova",
        "posthoc",
        "marginal_means",
        "mixed_effects",
    ]
    assert config.posthoc_methods == ("games_howell",)
    assert config.marginal_p_adjust == "none"


def test_config_rejects_invalid_posthoc_method():
    with pytest.raises(ValueError):
        AnalysisConfig(posthoc_methods=("magic",))


def test_config_rejects_negative_permutations():
    with pytest.raises(ValueError):
        AnalysisConfig(permanova_permutations=-1)


def test_specialized_controls_are_validated():
    with pytest.raises(ValueError):
        AnalysisConfig(
            enabled_blocks=("representation",),
            representation_distance_similarity_method="bad",
        )
    with pytest.raises(ValueError):
        AnalysisConfig(enabled_blocks=("anomaly",), anomaly_methods=("bad",))
    with pytest.raises(ValueError):
        AnalysisConfig(enabled_blocks=("bayesian",), bayesian_draws=20)
