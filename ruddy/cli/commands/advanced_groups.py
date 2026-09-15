"""CLI commands for Phase 9B advanced group/inference capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ruddy.bivariate import analyze_posthoc
from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.cli.commands.projections import load_feature_matrix
from ruddy.core.io import write_json, write_table
from ruddy.factorial import analyze_marginal_means, analyze_mixed_effects
from ruddy.multivariate import analyze_permutation_group_structure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ruddy.bivariate import PosthocResult
    from ruddy.cli._options import CliArgs
    from ruddy.factorial import MarginalMeansResult, MixedEffectsResult
    from ruddy.multivariate import PermutationGroupResult


def write_permanova_result(result: PermutationGroupResult, output_dir: str | Path) -> Path:
    """Persist structured PERMANOVA and PERMDISP artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.summary, target / "permutation_group_summary.csv")
    write_table(result.groups, target / "permutation_group_levels.csv")
    write_table(result.distances_to_centroid, target / "permdisp_distances.csv")
    write_table(result.exclusions, target / "permutation_group_exclusions.csv")
    write_json(result.alignment.to_dict(), target / "permutation_group_alignment.json")
    write_json(result.provenance.to_dict(), target / "permutation_group_provenance.json")
    if result.advisories:
        write_json(
            {"advisories": [a.to_dict() for a in result.advisories]},
            target / "permutation_group_advisories.json",
        )
    return target


def write_posthoc_result(result: PosthocResult, output_dir: str | Path) -> Path:
    """Persist structured post-hoc comparison artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.comparisons, target / "posthoc_comparisons.csv")
    write_json(result.provenance.to_dict(), target / "posthoc_provenance.json")
    return target


def write_marginal_means_result(result: MarginalMeansResult, output_dir: str | Path) -> Path:
    """Persist structured marginal-means artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.means, target / "marginal_means.csv")
    write_table(result.contrasts, target / "marginal_contrasts.csv")
    write_table(result.exclusions, target / "marginal_exclusions.csv")
    write_json(result.provenance.to_dict(), target / "marginal_means_provenance.json")
    return target


def write_mixed_effects_result(result: MixedEffectsResult, output_dir: str | Path) -> Path:
    """Persist structured mixed-effects artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.fixed_effects, target / "mixed_fixed_effects.csv")
    write_table(result.variance_components, target / "mixed_variance_components.csv")
    write_table(result.random_effects, target / "mixed_random_effects.csv")
    write_table(result.exclusions, target / "mixed_exclusions.csv")
    write_json(result.model_summary, target / "mixed_model_summary.json")
    write_json(result.provenance.to_dict(), target / "mixed_provenance.json")
    if result.advisories:
        write_json(
            {"advisories": [a.to_dict() for a in result.advisories]},
            target / "mixed_advisories.json",
        )
    return target


def run_permanova(args: CliArgs) -> int:
    """Run the ``ruddy model permanova`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    feature_source = args.feature_input or args.input
    features = load_feature_matrix(
        feature_source,
        id_column=args.feature_id_column or args.id_column,
        feature_columns=tuple(args.feature_column or ()),
        ids_file=args.feature_ids_file,
    )
    factor = args.permanova_factor or (
        args.group[0] if getattr(args, "group", None) and len(args.group) == 1 else None
    )
    if factor is None:
        msg = "PERMANOVA CLI requires --permanova-factor or exactly one --group."
        raise ValueError(msg)
    result = analyze_permutation_group_structure(
        features, dataset, factor=factor, **config.permanova_kwargs()
    )
    if args.output_dir:
        print(write_permanova_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "status": result.status.value,
                    "reason": result.reason,
                    "rows": len(result.summary),
                },
                indent=2,
            )
        )
    return 0


def run_posthoc(args: CliArgs) -> int:
    """Run the ``ruddy analyze posthoc`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_posthoc(
        dataset,
        response=args.posthoc_response,
        factor=args.posthoc_factor,
        **config.posthoc_kwargs(),
    )
    if args.output_dir:
        print(write_posthoc_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "rows": len(result.comparisons),
                    "methods": sorted(result.comparisons.method.unique().tolist())
                    if len(result.comparisons)
                    else [],
                },
                indent=2,
            )
        )
    return 0


def _parse_interactions(values: Sequence[str] | None) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(part.strip() for part in value.split(":")) for value in (values or ()))


def _parse_terms(values: Sequence[str] | None) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(part.strip() for part in value.split(":")) for value in (values or ()))


def run_marginal_means(args: CliArgs) -> int:
    """Run the ``ruddy model marginal-means`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    if args.marginal_formula:
        result = analyze_marginal_means(
            dataset,
            formula=args.marginal_formula,
            terms=_parse_terms(args.marginal_term) or None,
            **config.marginal_means_kwargs(),
        )
    else:
        result = analyze_marginal_means(
            dataset,
            response=args.marginal_response,
            factors=tuple(args.marginal_factor or ()),
            covariates=tuple(args.marginal_covariate or ()),
            interactions=_parse_interactions(args.marginal_interaction),
            terms=_parse_terms(args.marginal_term) or None,
            **config.marginal_means_kwargs(),
        )
    if args.output_dir:
        print(write_marginal_means_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "mean_rows": len(result.means),
                    "contrast_rows": len(result.contrasts),
                },
                indent=2,
            )
        )
    return 0


def run_mixed_effects(args: CliArgs) -> int:
    """Run the ``ruddy model mixed-effects`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    kwargs = config.mixed_effects_kwargs()
    if args.mixed_formula:
        result = analyze_mixed_effects(dataset, group=args.mixed_group, formula=args.mixed_formula, **kwargs)
    else:
        result = analyze_mixed_effects(
            dataset,
            group=args.mixed_group,
            response=args.mixed_response,
            factors=tuple(args.mixed_factor or ()),
            covariates=tuple(args.mixed_covariate or ()),
            interactions=_parse_interactions(args.mixed_interaction),
            **kwargs,
        )
    if args.output_dir:
        print(write_mixed_effects_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "status": result.status.value,
                    "reason": result.reason,
                    **result.model_summary,
                },
                indent=2,
                default=str,
            )
        )
    return 0


__all__ = [
    "run_marginal_means",
    "run_mixed_effects",
    "run_permanova",
    "run_posthoc",
    "write_marginal_means_result",
    "write_mixed_effects_result",
    "write_permanova_result",
    "write_posthoc_result",
]
