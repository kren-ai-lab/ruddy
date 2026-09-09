"""CLI commands for Phase 9B advanced group/inference capabilities."""

from __future__ import annotations

import json
from pathlib import Path

from ruddy.bivariate import analyze_posthoc
from ruddy.factorial import analyze_marginal_means, analyze_mixed_effects
from ruddy.multivariate import analyze_permutation_group_structure
from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.cli.commands.projections import load_feature_matrix


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_permanova_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(target / "permutation_group_summary.csv", index=False)
    result.groups.to_csv(target / "permutation_group_levels.csv", index=False)
    result.distances_to_centroid.to_csv(target / "permdisp_distances.csv", index=False)
    result.exclusions.to_csv(target / "permutation_group_exclusions.csv", index=False)
    _write_json(target / "permutation_group_alignment.json", result.alignment.to_dict())
    _write_json(target / "permutation_group_provenance.json", result.provenance.to_dict())
    if result.advisories:
        _write_json(target / "permutation_group_advisories.json", {"advisories": [a.to_dict() for a in result.advisories]})
    return target


def write_posthoc_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.comparisons.to_csv(target / "posthoc_comparisons.csv", index=False)
    _write_json(target / "posthoc_provenance.json", result.provenance.to_dict())
    return target


def write_marginal_means_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.means.to_csv(target / "marginal_means.csv", index=False)
    result.contrasts.to_csv(target / "marginal_contrasts.csv", index=False)
    result.exclusions.to_csv(target / "marginal_exclusions.csv", index=False)
    _write_json(target / "marginal_means_provenance.json", result.provenance.to_dict())
    return target


def write_mixed_effects_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.fixed_effects.to_csv(target / "mixed_fixed_effects.csv", index=False)
    result.variance_components.to_csv(target / "mixed_variance_components.csv", index=False)
    result.random_effects.to_csv(target / "mixed_random_effects.csv", index=False)
    result.exclusions.to_csv(target / "mixed_exclusions.csv", index=False)
    _write_json(target / "mixed_model_summary.json", result.model_summary)
    _write_json(target / "mixed_provenance.json", result.provenance.to_dict())
    if result.advisories:
        _write_json(target / "mixed_advisories.json", {"advisories": [a.to_dict() for a in result.advisories]})
    return target


def run_permanova(args) -> int:
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    feature_source = args.feature_input or args.input
    features = load_feature_matrix(
        feature_source,
        id_column=args.feature_id_column or args.id_column,
        feature_columns=tuple(args.feature_column or ()),
        ids_file=args.feature_ids_file,
    )
    factor = args.permanova_factor or (args.group[0] if getattr(args, "group", None) and len(args.group) == 1 else None)
    if factor is None:
        raise ValueError("PERMANOVA CLI requires --permanova-factor or exactly one --group.")
    result = analyze_permutation_group_structure(features, dataset, factor=factor, **config.permanova_kwargs())
    if args.output_dir:
        print(write_permanova_result(result, args.output_dir))
    else:
        print(json.dumps({"status": result.status.value, "reason": result.reason, "rows": len(result.summary)}, indent=2))
    return 0


def run_posthoc(args) -> int:
    config = build_config(args); dataset = load_dataset(args.input, config)
    result = analyze_posthoc(dataset, response=args.posthoc_response, factor=args.posthoc_factor, **config.posthoc_kwargs())
    if args.output_dir:
        print(write_posthoc_result(result, args.output_dir))
    else:
        print(json.dumps({"rows": len(result.comparisons), "methods": sorted(result.comparisons.method.unique().tolist()) if len(result.comparisons) else []}, indent=2))
    return 0


def _parse_interactions(values) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(part.strip() for part in value.split(":")) for value in (values or ()))


def _parse_terms(values) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(part.strip() for part in value.split(":")) for value in (values or ()))


def run_marginal_means(args) -> int:
    config = build_config(args); dataset = load_dataset(args.input, config)
    if args.marginal_formula:
        result = analyze_marginal_means(
            dataset, formula=args.marginal_formula, terms=_parse_terms(args.marginal_term) or None,
            **config.marginal_means_kwargs(),
        )
    else:
        result = analyze_marginal_means(
            dataset, response=args.marginal_response,
            factors=tuple(args.marginal_factor or ()),
            covariates=tuple(args.marginal_covariate or ()),
            interactions=_parse_interactions(args.marginal_interaction),
            terms=_parse_terms(args.marginal_term) or None,
            **config.marginal_means_kwargs(),
        )
    if args.output_dir:
        print(write_marginal_means_result(result, args.output_dir))
    else:
        print(json.dumps({"mean_rows": len(result.means), "contrast_rows": len(result.contrasts)}, indent=2))
    return 0


def run_mixed_effects(args) -> int:
    config = build_config(args); dataset = load_dataset(args.input, config)
    kwargs = config.mixed_effects_kwargs()
    if args.mixed_formula:
        result = analyze_mixed_effects(dataset, group=args.mixed_group, formula=args.mixed_formula, **kwargs)
    else:
        result = analyze_mixed_effects(
            dataset, group=args.mixed_group, response=args.mixed_response,
            factors=tuple(args.mixed_factor or ()), covariates=tuple(args.mixed_covariate or ()),
            interactions=_parse_interactions(args.mixed_interaction), **kwargs,
        )
    if args.output_dir:
        print(write_mixed_effects_result(result, args.output_dir))
    else:
        print(json.dumps({"status": result.status.value, "reason": result.reason, **result.model_summary}, indent=2, default=str))
    return 0


__all__ = [
    "run_permanova", "run_posthoc", "run_marginal_means", "run_mixed_effects",
    "write_permanova_result", "write_posthoc_result", "write_marginal_means_result", "write_mixed_effects_result",
]
