import pandas as pd
import polars.testing as pl_testing
import pytest

from ruddy import AnalysisConfig, analyze
from ruddy.bivariate import analyze_posthoc
from ruddy.factorial import analyze_marginal_means, analyze_mixed_effects
from ruddy.multivariate import analyze_permutation_group_structure


def test_unified_permanova_matches_standalone(group_dataset, separated_features):
    config = AnalysisConfig(
        enabled_blocks=("permanova",),
        groups=("group",),
        permanova_permutations=19,
        random_state=4,
    )
    unified = analyze(group_dataset, config=config, features=separated_features)
    standalone = analyze_permutation_group_structure(
        separated_features, group_dataset, factor="group", n_permutations=19, random_state=4
    )
    assert unified.permanova is not None
    pl_testing.assert_frame_equal(unified.permanova.summary, standalone.summary)


def test_unified_permanova_requires_features(group_dataset):
    config = AnalysisConfig(enabled_blocks=("permanova",), groups=("group",))
    with pytest.raises(ValueError, match="FeatureMatrix"):
        analyze(group_dataset, config=config)


def test_unified_posthoc_matches_standalone(group_dataset):
    config = AnalysisConfig(
        enabled_blocks=("posthoc",),
        responses=("y",),
        groups=("group",),
        posthoc_methods=("tukey_hsd",),
        min_group_n=2,
    )
    unified = analyze(group_dataset, config=config)
    standalone = analyze_posthoc(
        group_dataset, response="y", factor="group", methods=("tukey_hsd",), min_group_n=2
    )
    assert unified.posthoc is not None
    pl_testing.assert_frame_equal(unified.posthoc.comparisons, standalone.comparisons)


def test_unified_marginal_means_matches_standalone(group_dataset):
    config = AnalysisConfig(
        enabled_blocks=("marginal_means",),
        responses=("y",),
        marginal_factors=("group",),
        marginal_covariates=("x",),
    )
    unified = analyze(group_dataset, config=config)
    standalone = analyze_marginal_means(group_dataset, response="y", factors=("group",), covariates=("x",))
    assert unified.marginal_means is not None
    pd.testing.assert_frame_equal(unified.marginal_means.means, standalone.means)
    pd.testing.assert_frame_equal(unified.marginal_means.contrasts, standalone.contrasts)


def test_unified_mixed_effects_matches_standalone(random_intercept_dataset):
    config = AnalysisConfig(
        enabled_blocks=("mixed_effects",),
        responses=("y",),
        mixed_group="batch",
        mixed_factors=("condition",),
        mixed_covariates=("x",),
        mixed_reml=False,
    )
    unified = analyze(random_intercept_dataset, config=config)
    standalone = analyze_mixed_effects(
        random_intercept_dataset,
        group="batch",
        response="y",
        factors=("condition",),
        covariates=("x",),
        reml=False,
    )
    assert unified.mixed_effects is not None
    assert unified.mixed_effects.status == standalone.status
    pd.testing.assert_frame_equal(unified.mixed_effects.fixed_effects, standalone.fixed_effects)


def test_unified_mixed_effects_requires_random_group(random_intercept_dataset):
    config = AnalysisConfig(enabled_blocks=("mixed_effects",), responses=("y",), mixed_factors=("condition",))
    with pytest.raises(ValueError, match="mixed_group"):
        analyze(random_intercept_dataset, config=config)
