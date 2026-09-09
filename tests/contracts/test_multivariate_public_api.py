from __future__ import annotations

import ruddy


def test_multivariate_public_api_is_exposed():
    expected = {
        "analyze_covariance_structure",
        "analyze_collinearity",
        "analyze_mahalanobis",
        "analyze_manova",
        "analyze_multivariate",
        "CovarianceResult",
        "CollinearityResult",
        "MahalanobisResult",
        "MANOVAResult",
        "MultivariateResult",
    }
    assert expected <= set(ruddy.__all__)
