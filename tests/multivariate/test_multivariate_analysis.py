from __future__ import annotations

from ruddy import ResultStatus, analyze_multivariate


def test_combined_multivariate_result(well_conditioned_features):
    result = analyze_multivariate(well_conditioned_features, random_state=9)
    assert result.status is ResultStatus.OK
    assert result.covariance.status is ResultStatus.OK
    assert result.collinearity.status is ResultStatus.OK
    assert result.mahalanobis.status is ResultStatus.OK
    assert result.provenance.analysis == "multivariate"
