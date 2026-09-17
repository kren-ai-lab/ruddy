"""Unified result graph returned by :func:`ruddy.analyze`."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ruddy.core.enums import AnalysisBlock

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ruddy.analysis.intervals import ConfidenceIntervalResult
    from ruddy.anomaly import AnomalyResult
    from ruddy.bayesian import BayesianEDAResult
    from ruddy.bivariate import (
        BivariateResult,
        ContingencyDiagnosticsResult,
        DependenceResult,
        GroupAnalysisResult,
        PosthocResult,
    )
    from ruddy.compositional import CompositionalResult
    from ruddy.data.validation import AlignmentReport
    from ruddy.factorial import FactorialResult, MarginalMeansResult, MixedEffectsResult
    from ruddy.multivariate import MANOVAResult, MultivariateResult, PermutationGroupResult
    from ruddy.profiling import ProfilingResult
    from ruddy.projections import PCAResult, ProjectionResult
    from ruddy.representation import RepresentationComparisonResult
    from ruddy.results import AnalysisProvenance
    from ruddy.univariate import DistributionDiagnosticsResult, OutlierResult, UnivariateResult


@dataclass(frozen=True, slots=True)
class UnifiedAnalysisResult:
    """Composable graph of explicitly requested Ruddy analysis components."""

    profiling: ProfilingResult | None = None
    univariate: UnivariateResult | None = None
    bivariate: BivariateResult | None = None
    groups: GroupAnalysisResult | None = None
    outliers: OutlierResult | None = None
    pca: PCAResult | None = None
    tsne: ProjectionResult | None = None
    umap: ProjectionResult | None = None
    multivariate: MultivariateResult | None = None
    manova: MANOVAResult | None = None
    factorial: FactorialResult | None = None
    diagnostics: DistributionDiagnosticsResult | None = None
    dependence: DependenceResult | None = None
    contingency: ContingencyDiagnosticsResult | None = None
    intervals: ConfidenceIntervalResult | None = None
    permanova: PermutationGroupResult | None = None
    posthoc: PosthocResult | None = None
    marginal_means: MarginalMeansResult | None = None
    mixed_effects: MixedEffectsResult | None = None
    representation: RepresentationComparisonResult | None = None
    compositional: CompositionalResult | None = None
    bayesian: BayesianEDAResult | None = None
    anomaly: AnomalyResult | None = None
    feature_alignment: AlignmentReport | None = None
    executed_blocks: tuple[AnalysisBlock, ...] = ()
    provenance: AnalysisProvenance | None = None

    def __post_init__(self) -> None:
        blocks = tuple(
            block if isinstance(block, AnalysisBlock) else AnalysisBlock(block)
            for block in self.executed_blocks
        )
        if len(set(blocks)) != len(blocks):
            msg = "executed_blocks cannot contain duplicates."
            raise ValueError(msg)
        object.__setattr__(self, "executed_blocks", blocks)

    def component(self, block: AnalysisBlock | str) -> Any | None:
        """Return one named component without coupling callers to field dispatch."""
        resolved = block if isinstance(block, AnalysisBlock) else AnalysisBlock(block)
        return getattr(self, resolved.value)

    @property
    def components(self) -> Mapping[AnalysisBlock, Any]:
        """Return an immutable mapping containing only executed components."""
        return MappingProxyType({block: self.component(block) for block in self.executed_blocks})

    def summary(self) -> dict[str, Any]:
        """Return a lightweight serialization-safe execution summary."""
        return {
            "executed_blocks": [block.value for block in self.executed_blocks],
            "components": {block.value: self.component(block) is not None for block in self.executed_blocks},
            "feature_alignment": (
                None if self.feature_alignment is None else self.feature_alignment.to_dict()
            ),
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
        }
