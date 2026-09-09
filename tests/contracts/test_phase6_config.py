import pytest

from ruddy import AnalysisConfig, ScalingMethod


def test_phase6_config_defaults_do_not_scale():
    config = AnalysisConfig()
    assert config.projection_scaling is ScalingMethod.NONE
    assert config.pca_kwargs()["scaling"] is ScalingMethod.NONE
    assert config.projection_n_components == 2


def test_phase6_config_projection_kwargs():
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
def test_phase6_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        AnalysisConfig(**kwargs)
