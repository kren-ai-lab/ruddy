"""Orchestration for Ruddy mixed-type bivariate analysis."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ruddy.bivariate.associations import summarize_categorical_associations
from ruddy.bivariate.comparisons import summarize_numeric_categorical_comparisons
from ruddy.bivariate.correlations import summarize_correlations
from ruddy.core.enums import ComparisonTest, CorrelationMethod, PAdjustMethod
from ruddy.data import TabularDataset
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class BivariateResult:
    """Complete mixed-type bivariate output."""

    correlations: pl.DataFrame
    comparisons: pl.DataFrame
    categorical_associations: pl.DataFrame
    provenance: AnalysisProvenance


def analyze_bivariate(
    dataset: TabularDataset,
    *,
    correlations: tuple[CorrelationMethod | str, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
        CorrelationMethod.KENDALL,
    ),
    comparison_tests: tuple[ComparisonTest | str, ...] = (
        ComparisonTest.WELCH_T,
        ComparisonTest.MANN_WHITNEY,
        ComparisonTest.WELCH_ANOVA,
        ComparisonTest.KRUSKAL_WALLIS,
        ComparisonTest.CHI_SQUARE,
        ComparisonTest.FISHER_EXACT,
    ),
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    min_group_n: int = 3,
    min_correlation_pairs: int = 3,
    max_group_levels: int = 20,
    max_correlation_columns: int = 100,
    max_category_levels: int = 50,
    pairwise: bool = False,
) -> BivariateResult:
    """Run the complete domain-agnostic bivariate layer."""
    numeric_tests = tuple(
        test
        for test in comparison_tests
        if ComparisonTest(test)
        in {
            ComparisonTest.WELCH_T,
            ComparisonTest.MANN_WHITNEY,
            ComparisonTest.WELCH_ANOVA,
            ComparisonTest.KRUSKAL_WALLIS,
        }
    )
    categorical_tests = tuple(
        test
        for test in comparison_tests
        if ComparisonTest(test) in {ComparisonTest.CHI_SQUARE, ComparisonTest.FISHER_EXACT}
    )
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)

    correlation_table = summarize_correlations(
        dataset,
        methods=correlations,
        min_complete_pairs=min_correlation_pairs,
        max_columns=max_correlation_columns,
        p_adjust=correction,
    )
    comparison_table = summarize_numeric_categorical_comparisons(
        dataset,
        tests=numeric_tests,
        min_group_n=min_group_n,
        max_group_levels=max_group_levels,
        pairwise=pairwise,
        p_adjust=correction,
    )
    association_table = summarize_categorical_associations(
        dataset,
        tests=categorical_tests,
        max_category_levels=max_category_levels,
        p_adjust=correction,
    )
    provenance = AnalysisProvenance(
        analysis="bivariate",
        parameters={
            "correlations": tuple(CorrelationMethod(method).value for method in correlations),
            "comparison_tests": tuple(ComparisonTest(test).value for test in comparison_tests),
            "p_adjust": correction.value,
            "min_group_n": min_group_n,
            "min_correlation_pairs": min_correlation_pairs,
            "max_group_levels": max_group_levels,
            "max_correlation_columns": max_correlation_columns,
            "max_category_levels": max_category_levels,
            "pairwise": pairwise,
            "numeric_pair_policy": "finite_pairwise_complete",
            "comparison_test_selection": "explicit_by_group_count_not_normality",
            "categorical_low_expected_count_policy": "advisory_no_automatic_test_switch",
        },
        input_summary={
            "n_observations": dataset.frame.height,
            "n_columns": dataset.frame.width,
            "id_column": dataset.id_column,
        },
    )
    return BivariateResult(
        correlations=correlation_table,
        comparisons=comparison_table,
        categorical_associations=association_table,
        provenance=provenance,
    )
