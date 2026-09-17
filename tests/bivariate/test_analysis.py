from __future__ import annotations

from ruddy import BivariateResult, analyze_bivariate


def test_analyze_bivariate_returns_all_three_tables(mixed_dataset) -> None:
    result = analyze_bivariate(mixed_dataset)
    assert isinstance(result, BivariateResult)
    assert not result.correlations.empty
    assert not result.comparisons.empty
    assert not result.categorical_associations.empty
    assert result.provenance.analysis == "bivariate"
    assert (
        result.provenance.parameters["comparison_test_selection"] == "explicit_by_group_count_not_normality"
    )
