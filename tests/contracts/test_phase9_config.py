import pytest

from ruddy import AnalysisBlock, AnalysisConfig


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
    assert AnalysisConfig(feature_alignment="partial").feature_alignment == "partial"


def test_factorial_interactions_are_immutable_tuples():
    config = AnalysisConfig(factorial_interactions=(("A", "B"),))
    assert config.factorial_interactions == (("A", "B"),)


def test_factorial_interaction_requires_two_predictors():
    with pytest.raises(ValueError, match="at least two"):
        AnalysisConfig(factorial_interactions=(("A",),))
