"""Unified Ruddy analysis command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ruddy.analysis import analyze
from ruddy.cli._console import record_blocks
from ruddy.cli.commands.advanced_groups import (
    write_marginal_means_result,
    write_mixed_effects_result,
    write_permanova_result,
    write_posthoc_result,
)
from ruddy.cli.commands.bivariate import write_bivariate_result
from ruddy.cli.commands.descriptive import (
    build_config,
    load_dataset,
    write_profiling_result,
    write_univariate_result,
)
from ruddy.cli.commands.factorial import write_factorial_result
from ruddy.cli.commands.groups import write_group_result
from ruddy.cli.commands.inference import (
    write_confidence_interval_result,
    write_contingency_diagnostics_result,
    write_dependence_result,
    write_distribution_diagnostics_result,
)
from ruddy.cli.commands.multivariate import (
    write_manova_result,
    write_multivariate_result,
)
from ruddy.cli.commands.outliers import write_outlier_result
from ruddy.cli.commands.projections import (
    load_feature_matrix,
    write_pca_result,
    write_projection_result,
)
from ruddy.cli.commands.specialized import (
    write_anomaly_result,
    write_bayesian_result,
    write_compositional_result,
    write_representation_result,
)
from ruddy.core import AnalysisBlock, ResultStatus
from ruddy.core.io import write_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruddy.analysis import UnifiedAnalysisResult
    from ruddy.cli._options import CliArgs


def write_unified_result(result: UnifiedAnalysisResult, output_dir: str | Path) -> Path:
    """Persist each executed component in its own stable subdirectory."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    writers = {
        AnalysisBlock.PROFILING: write_profiling_result,
        AnalysisBlock.UNIVARIATE: write_univariate_result,
        AnalysisBlock.BIVARIATE: write_bivariate_result,
        AnalysisBlock.GROUPS: write_group_result,
        AnalysisBlock.OUTLIERS: write_outlier_result,
        AnalysisBlock.PCA: write_pca_result,
        AnalysisBlock.TSNE: write_projection_result,
        AnalysisBlock.UMAP: write_projection_result,
        AnalysisBlock.MULTIVARIATE: write_multivariate_result,
        AnalysisBlock.MANOVA: write_manova_result,
        AnalysisBlock.FACTORIAL: write_factorial_result,
        AnalysisBlock.DIAGNOSTICS: write_distribution_diagnostics_result,
        AnalysisBlock.DEPENDENCE: write_dependence_result,
        AnalysisBlock.CONTINGENCY: write_contingency_diagnostics_result,
        AnalysisBlock.INTERVALS: write_confidence_interval_result,
        AnalysisBlock.PERMANOVA: write_permanova_result,
        AnalysisBlock.POSTHOC: write_posthoc_result,
        AnalysisBlock.MARGINAL_MEANS: write_marginal_means_result,
        AnalysisBlock.MIXED_EFFECTS: write_mixed_effects_result,
        AnalysisBlock.REPRESENTATION: write_representation_result,
        AnalysisBlock.COMPOSITIONAL: write_compositional_result,
        AnalysisBlock.BAYESIAN: write_bayesian_result,
        AnalysisBlock.ANOMALY: write_anomaly_result,
    }
    for block in result.executed_blocks:
        component = result.component(block)
        if component is not None:
            writers[block](component, target / block.value)
    write_json(result.summary(), target / "analysis_summary.json")
    return target


def _block_ok(component: object) -> bool:
    # Not every block result carries a status; those that do not always ran.
    return component is not None and getattr(component, "status", ResultStatus.OK) is ResultStatus.OK


def run_analyze(args: CliArgs) -> int:
    """Run the ``ruddy analyze run`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    feature_blocks = {
        AnalysisBlock.PCA,
        AnalysisBlock.TSNE,
        AnalysisBlock.UMAP,
        AnalysisBlock.MULTIVARIATE,
        AnalysisBlock.PERMANOVA,
        AnalysisBlock.REPRESENTATION,
        AnalysisBlock.COMPOSITIONAL,
        AnalysisBlock.ANOMALY,
    }
    features = None
    if any(block in feature_blocks for block in config.enabled_blocks):
        feature_source = args.feature_input or args.input
        features = load_feature_matrix(
            feature_source,
            id_column=args.feature_id_column or args.id_column,
            feature_columns=tuple(args.feature_column or ()),
            ids_file=args.feature_ids_file,
        )
    comparison_features = None
    if AnalysisBlock.REPRESENTATION in config.enabled_blocks:
        if not args.comparison_feature_input:
            msg = "--comparison-feature-input is required when enabling representation."
            raise ValueError(msg)
        comparison_features = load_feature_matrix(
            args.comparison_feature_input,
            id_column=args.comparison_feature_id_column,
            feature_columns=tuple(args.comparison_feature_column or ()),
            ids_file=args.comparison_feature_ids_file,
        )
    result = analyze(
        dataset,
        config=config,
        features=features,
        comparison_features=comparison_features,
    )
    record_blocks((block.value, _block_ok(result.component(block))) for block in result.executed_blocks)
    if args.output_dir:
        print(write_unified_result(result, args.output_dir))
    else:
        print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0


__all__ = ["run_analyze", "write_unified_result"]
