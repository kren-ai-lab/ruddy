"""CLI commands for Phase 9A EDA-completeness capabilities."""

from __future__ import annotations

import json
from pathlib import Path

from ruddy.analysis import analyze_confidence_intervals
from ruddy.bivariate import analyze_contingency_diagnostics, analyze_dependence
from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.univariate import analyze_distribution_diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_distribution_diagnostics_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.normality.to_csv(target / "normality_diagnostics.csv", index=False)
    result.dispersion.to_csv(target / "dispersion_diagnostics.csv", index=False)
    _write_json(target / "diagnostics_provenance.json", result.provenance.to_dict())
    return target


def write_dependence_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.partial_correlations.to_csv(target / "partial_correlations.csv", index=False)
    result.distance_correlations.to_csv(target / "distance_correlations.csv", index=False)
    result.mutual_information.to_csv(target / "mutual_information.csv", index=False)
    _write_json(target / "dependence_provenance.json", result.provenance.to_dict())
    return target


def write_contingency_diagnostics_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(target / "contingency_summary.csv", index=False)
    result.cells.to_csv(target / "contingency_cells.csv", index=False)
    _write_json(target / "contingency_provenance.json", result.provenance.to_dict())
    return target


def write_confidence_interval_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.means.to_csv(target / "mean_intervals.csv", index=False)
    result.correlations.to_csv(target / "correlation_intervals.csv", index=False)
    result.mean_differences.to_csv(target / "mean_difference_intervals.csv", index=False)
    result.effect_sizes.to_csv(target / "effect_size_intervals.csv", index=False)
    result.odds_ratios.to_csv(target / "odds_ratio_intervals.csv", index=False)
    _write_json(target / "interval_provenance.json", result.provenance.to_dict())
    return target


def run_diagnostics(args) -> int:
    config = build_config(args); dataset = load_dataset(args.input, config)
    result = analyze_distribution_diagnostics(dataset, **config.distribution_diagnostics_kwargs())
    if args.output_dir:
        print(write_distribution_diagnostics_result(result, args.output_dir))
    else:
        print(json.dumps({"normality_rows": len(result.normality), "dispersion_rows": len(result.dispersion)}, indent=2))
    return 0


def run_dependence(args) -> int:
    config = build_config(args); dataset = load_dataset(args.input, config)
    result = analyze_dependence(dataset, **config.dependence_kwargs())
    if args.output_dir:
        print(write_dependence_result(result, args.output_dir))
    else:
        print(json.dumps({"partial_rows": len(result.partial_correlations), "distance_rows": len(result.distance_correlations), "mi_rows": len(result.mutual_information)}, indent=2))
    return 0


def run_contingency(args) -> int:
    config = build_config(args); dataset = load_dataset(args.input, config)
    result = analyze_contingency_diagnostics(dataset, **config.contingency_kwargs())
    if args.output_dir:
        print(write_contingency_diagnostics_result(result, args.output_dir))
    else:
        print(json.dumps({"table_rows": len(result.summary), "cell_rows": len(result.cells)}, indent=2))
    return 0


def run_intervals(args) -> int:
    config = build_config(args); dataset = load_dataset(args.input, config)
    result = analyze_confidence_intervals(dataset, **config.interval_kwargs())
    if args.output_dir:
        print(write_confidence_interval_result(result, args.output_dir))
    else:
        print(json.dumps({
            "mean_rows": len(result.means),
            "correlation_rows": len(result.correlations),
            "mean_difference_rows": len(result.mean_differences),
            "effect_size_rows": len(result.effect_sizes),
            "odds_ratio_rows": len(result.odds_ratios),
        }, indent=2))
    return 0


__all__ = [
    "run_contingency", "run_dependence", "run_diagnostics", "run_intervals",
    "write_confidence_interval_result", "write_contingency_diagnostics_result",
    "write_dependence_result", "write_distribution_diagnostics_result",
]
