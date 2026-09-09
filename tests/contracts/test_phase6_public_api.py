import ruddy


def test_phase6_public_api_is_exposed():
    for name in (
        "PCAResult",
        "ProjectionResult",
        "ScalingMethod",
        "ProjectionMethod",
        "analyze_pca",
        "analyze_tsne",
        "analyze_umap",
        "prepare_features",
    ):
        assert hasattr(ruddy, name)
