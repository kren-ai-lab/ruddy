"""CLI commands for Phase 9A EDA-completeness capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ruddy.analysis import analyze_confidence_intervals
from ruddy.bivariate import analyze_contingency_diagnostics, analyze_dependence
from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.core.io import write_json, write_table
from ruddy.univariate import analyze_distribution_diagnostics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruddy.analysis import ConfidenceIntervalResult
    from ruddy.bivariate import ContingencyDiagnosticsResult, DependenceResult
    from ruddy.cli._options import CliArgs
    from ruddy.univariate import DistributionDiagnosticsResult


def write_distribution_diagnostics_result(
    result: DistributionDiagnosticsResult, output_dir: str | Path
) -> Path:
    """Persist structured distribution-diagnostics artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.normality, target / "normality_diagnostics.csv")
    write_table(result.dispersion, target / "dispersion_diagnostics.csv")
    write_json(result.provenance.to_dict(), target / "diagnostics_provenance.json")
    return target


def write_dependence_result(result: DependenceResult, output_dir: str | Path) -> Path:
    """Persist structured dependence artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.partial_correlations, target / "partial_correlations.csv")
    write_table(result.distance_correlations, target / "distance_correlations.csv")
    write_table(result.mutual_information, target / "mutual_information.csv")
    write_json(result.provenance.to_dict(), target / "dependence_provenance.json")
    return target


def write_contingency_diagnostics_result(
    result: ContingencyDiagnosticsResult, output_dir: str | Path
) -> Path:
    """Persist structured contingency-diagnostics artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.summary, target / "contingency_summary.csv")
    write_table(result.cells, target / "contingency_cells.csv")
    write_json(result.provenance.to_dict(), target / "contingency_provenance.json")
    return target


def write_confidence_interval_result(result: ConfidenceIntervalResult, output_dir: str | Path) -> Path:
    """Persist structured confidence-interval artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.means, target / "mean_intervals.csv")
    write_table(result.correlations, target / "correlation_intervals.csv")
    write_table(result.mean_differences, target / "mean_difference_intervals.csv")
    write_table(result.effect_sizes, target / "effect_size_intervals.csv")
    write_table(result.odds_ratios, target / "odds_ratio_intervals.csv")
    write_json(result.provenance.to_dict(), target / "interval_provenance.json")
    return target


def run_diagnostics(args: CliArgs) -> int:
    """Run the ``ruddy inspect diagnostics`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_distribution_diagnostics(dataset, **config.distribution_diagnostics_kwargs())
    if args.output_dir:
        print(write_distribution_diagnostics_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "normality_rows": len(result.normality),
                    "dispersion_rows": len(result.dispersion),
                },
                indent=2,
            )
        )
    return 0


def run_dependence(args: CliArgs) -> int:
    """Run the ``ruddy analyze dependence`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_dependence(dataset, **config.dependence_kwargs())
    if args.output_dir:
        print(write_dependence_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "partial_rows": len(result.partial_correlations),
                    "distance_rows": len(result.distance_correlations),
                    "mi_rows": len(result.mutual_information),
                },
                indent=2,
            )
        )
    return 0


def run_contingency(args: CliArgs) -> int:
    """Run the ``ruddy analyze contingency`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_contingency_diagnostics(dataset, **config.contingency_kwargs())
    if args.output_dir:
        print(write_contingency_diagnostics_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {"table_rows": len(result.summary), "cell_rows": len(result.cells)},
                indent=2,
            )
        )
    return 0


def run_intervals(args: CliArgs) -> int:
    """Run the ``ruddy analyze intervals`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_confidence_intervals(dataset, **config.interval_kwargs())
    if args.output_dir:
        print(write_confidence_interval_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "mean_rows": len(result.means),
                    "correlation_rows": len(result.correlations),
                    "mean_difference_rows": len(result.mean_differences),
                    "effect_size_rows": len(result.effect_sizes),
                    "odds_ratio_rows": len(result.odds_ratios),
                },
                indent=2,
            )
        )
    return 0


__all__ = [
    "run_contingency",
    "run_dependence",
    "run_diagnostics",
    "run_intervals",
    "write_confidence_interval_result",
    "write_contingency_diagnostics_result",
    "write_dependence_result",
    "write_distribution_diagnostics_result",
]
