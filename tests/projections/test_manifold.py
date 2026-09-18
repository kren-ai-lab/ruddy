import importlib.util

import numpy as np
import pytest

from ruddy import FeatureMatrix, OptionalDependencyError, analyze_tsne, analyze_umap


def _features():
    rng = np.random.default_rng(12)
    return FeatureMatrix(rng.normal(size=(18, 5)), observation_ids=[f"x{i}" for i in range(18)])


def test_tsne_is_deterministic_under_fixed_seed_and_noninferential():
    features = _features()
    first = analyze_tsne(features, perplexity=5.0, max_iter=250, random_state=19)
    second = analyze_tsne(features, perplexity=5.0, max_iter=250, random_state=19)
    np.testing.assert_allclose(
        first.coordinates.select(["component_1", "component_2"]).to_numpy(),
        second.coordinates.select(["component_1", "component_2"]).to_numpy(),
        rtol=0.0,
        atol=0.0,
    )
    assert first.exploratory is True
    assert first.inferential_allowed is False
    assert first.provenance.parameters["inferential_allowed"] is False


def test_tsne_excludes_nonfinite_rows_explicitly():
    x = np.arange(60.0).reshape(15, 4)
    x[4, 2] = np.nan
    result = analyze_tsne(
        FeatureMatrix(x, observation_ids=list(range(15))),
        perplexity=4.0,
        max_iter=250,
        random_state=1,
    )
    assert result.exclusions["observation_id"].to_list() == [4]
    assert 4 not in result.coordinates["observation_id"].to_list()


def test_tsne_requires_valid_perplexity():
    with pytest.raises(ValueError, match="smaller"):
        analyze_tsne(_features(), perplexity=18.0, max_iter=250)


def test_umap_optional_dependency_or_deterministic_output():
    if importlib.util.find_spec("umap") is None:
        with pytest.raises(OptionalDependencyError, match="umap-learn"):
            analyze_umap(_features(), n_neighbors=5)
        return
    first = analyze_umap(_features(), n_neighbors=5, random_state=9)
    second = analyze_umap(_features(), n_neighbors=5, random_state=9)
    np.testing.assert_allclose(
        first.coordinates.select(["component_1", "component_2"]).to_numpy(),
        second.coordinates.select(["component_1", "component_2"]).to_numpy(),
    )
    assert first.exploratory is True
    assert first.inferential_allowed is False
