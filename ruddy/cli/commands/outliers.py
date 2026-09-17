"""CLI command for univariate statistical outlier diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.core.io import write_json, write_table
from ruddy.univariate import OutlierResult, analyze_outliers

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruddy.cli._options import CliArgs


def write_outlier_result(result: OutlierResult, output_dir: str | Path) -> Path:
    """Persist structured outlier and quality outputs."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.summaries, target / "outlier_summaries.csv")
    write_table(result.quality, target / "numeric_quality.csv")
    if not result.flags.empty:
        write_table(result.flags, target / "outlier_flags.csv")
    write_json(result.provenance.to_dict(), target / "outlier_provenance.json")
    return target


def run_outliers(args: CliArgs) -> int:
    """Run the ``ruddy inspect outliers`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_outliers(dataset, **config.outlier_kwargs())
    if args.output_dir:
        print(write_outlier_result(result, args.output_dir))
    else:
        payload = {
            "numeric_variables": int(result.quality.shape[0]),
            "summary_rows": int(result.summaries.shape[0]),
            "flagged_records": int(result.flags.shape[0]),
            "methods": list(result.provenance.parameters["methods"]),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
