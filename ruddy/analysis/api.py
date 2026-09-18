"""Public high-level analysis API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ruddy.analysis.engine import analyze
from ruddy.bivariate.groups import GroupAnalysisResult, analyze_grouped_responses
from ruddy.core.enums import ComparisonTest, PAdjustMethod

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ruddy.data import AlignedAnnotations, TabularDataset

__all__ = ["analyze", "analyze_groups"]


def analyze_groups(
    dataset: TabularDataset,
    *,
    responses: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
    annotations: Iterable[AlignedAnnotations] = (),
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
    max_group_levels: int = 20,
    max_category_levels: int = 50,
    pairwise: bool = False,
) -> GroupAnalysisResult:
    """Run response-centric grouped EDA using statistical roles and data kinds."""
    return analyze_grouped_responses(
        dataset,
        responses=responses,
        groups=groups,
        annotations=annotations,
        comparison_tests=comparison_tests,
        p_adjust=p_adjust,
        min_group_n=min_group_n,
        max_group_levels=max_group_levels,
        max_category_levels=max_category_levels,
        pairwise=pairwise,
    )
