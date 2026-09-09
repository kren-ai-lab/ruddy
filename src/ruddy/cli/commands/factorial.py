"""CLI command for factorial ANOVA/ANCOVA analysis."""

from __future__ import annotations

import json
from pathlib import Path

from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.factorial import FactorialResult, analyze_factorial


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_interactions(values: list[str]) -> tuple[tuple[str, ...], ...]:
    interactions: list[tuple[str, ...]] = []
    for value in values:
        columns = tuple(part.strip() for part in str(value).split(":"))
        if len(columns) < 2 or any(not column for column in columns):
            raise ValueError(
                "--interaction must use colon-separated predictor names, e.g. factor_a:factor_b."
            )
        if columns not in interactions:
            interactions.append(columns)
    return tuple(interactions)


def write_factorial_result(result: FactorialResult, output_dir: str | Path) -> Path:
    """Persist structured factorial outputs without plotting or narrative reporting."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result.design_terms.to_csv(target / "factorial_design_terms.csv", index=False)
    result.effects.to_csv(target / "factorial_effects.csv", index=False)
    result.coefficients.to_csv(target / "factorial_coefficients.csv", index=False)
    result.diagnostics.to_csv(target / "factorial_diagnostics.csv", index=False)
    if not result.observation_diagnostics.empty:
        result.observation_diagnostics.to_csv(
            target / "factorial_observation_diagnostics.csv", index=False
        )
    if not result.cells.empty:
        result.cells.to_csv(target / "factorial_cells.csv", index=False)
    if not result.exclusions.empty:
        result.exclusions.to_csv(target / "factorial_exclusions.csv", index=False)
    _write_json(target / "factorial_model_summary.json", result.model_summary)
    _write_json(
        target / "factorial_provenance.json",
        {
            "status": result.status.value,
            "reason": result.reason,
            "advisories": [advisory.to_dict() for advisory in result.advisories],
            "provenance": result.provenance.to_dict(),
        },
    )
    return target


def run_factorial(args) -> int:
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    kwargs = {
        "ss_type": int(args.ss_type),
        "p_adjust": args.p_adjust,
        "robust_covariance": None if args.robust_covariance == "none" else args.robust_covariance,
        "min_cell_n": args.min_cell_n,
        "max_factor_levels": args.max_factor_levels,
        "max_design_cells": args.max_design_cells,
        "max_design_columns": args.max_design_columns,
        "max_interaction_order": args.max_interaction_order,
        "diagnostic_alpha": args.diagnostic_alpha,
        "condition_number_threshold": args.condition_number_threshold,
    }
    if args.formula:
        result = analyze_factorial(dataset, formula=args.formula, **kwargs)
    else:
        responses = tuple(args.response or ())
        if len(responses) != 1:
            raise ValueError(
                "Factorial CLI requires exactly one --response unless --formula is used."
            )
        result = analyze_factorial(
            dataset,
            response=responses[0],
            factors=tuple(args.factor or ()),
            covariates=tuple(args.covariate or ()),
            interactions=_parse_interactions(args.interaction or []),
            **kwargs,
        )

    if args.output_dir:
        path = write_factorial_result(result, args.output_dir)
        print(path)
    else:
        payload = {
            "status": result.status.value,
            "reason": result.reason,
            "resolved_formula": result.design.resolved_formula,
            "ss_type": result.design.ss_type,
            "n_effects": int(len(result.effects)),
            "n_complete_case": result.model_summary.get("n_complete_case"),
            "n_advisories": len(result.advisories),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = ["run_factorial", "write_factorial_result"]
