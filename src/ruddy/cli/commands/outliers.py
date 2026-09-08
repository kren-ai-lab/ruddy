"""CLI command for univariate statistical outlier diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from ruddy.univariate import OutlierResult, analyze_outliers
from ruddy.cli.commands.descriptive import build_config, load_dataset


def write_outlier_result(result: OutlierResult, output_dir: str | Path) -> Path:
    """Persist structured outlier and quality outputs."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result.summaries.to_csv(target / "outlier_summaries.csv", index=False)
    result.quality.to_csv(target / "numeric_quality.csv", index=False)
    if not result.flags.empty:
        result.flags.to_csv(target / "outlier_flags.csv", index=False)
    (target / "outlier_provenance.json").write_text(
        json.dumps(result.provenance.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def run_outliers(args) -> int:
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
