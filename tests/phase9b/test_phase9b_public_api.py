import ruddy


def test_phase9b_public_api_symbols_exist():
    for name in (
        "analyze_permutation_group_structure", "analyze_posthoc",
        "analyze_marginal_means", "analyze_mixed_effects",
        "PermutationGroupResult", "PosthocResult", "MarginalMeansResult", "MixedEffectsResult",
    ):
        assert hasattr(ruddy, name), name
