"""CLI helpers for Phase 3 mixed-type bivariate analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ruddy.bivariate import BivariateResult, analyze_bivariate
from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.core.io import write_json, write_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruddy.cli._options import CliArgs


def write_bivariate_result(result: BivariateResult, output_dir: str | Path) -> Path:
    """Persist structured Phase 3 artifacts."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.correlations, target / "correlations.csv")
    write_table(result.comparisons, target / "comparisons.csv")
    write_table(result.categorical_associations, target / "categorical_associations.csv")
    write_json(result.provenance.to_dict(), target / "bivariate_provenance.json")
    return target


def run_bivariate(args: CliArgs) -> int:
    """Run the ``ruddy analyze bivariate`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_bivariate(dataset, **config.bivariate_kwargs())
    if args.output_dir:
        print(write_bivariate_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "correlations": len(result.correlations),
                    "comparisons": len(result.comparisons),
                    "categorical_associations": len(result.categorical_associations),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0
