"""CLI helpers for response-centric grouped analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ruddy.bivariate import GroupAnalysisResult, analyze_grouped_responses
from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.core import ColumnKind, ColumnRole
from ruddy.core.io import read_table, write_json, write_table
from ruddy.data import align_annotation_source

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruddy.cli._options import CliArgs


def _annotation_roles(args: CliArgs) -> dict[str, ColumnRole]:
    roles: dict[str, ColumnRole] = {}
    for attr, role in (
        ("annotation_factor", ColumnRole.FACTOR),
        ("annotation_response", ColumnRole.RESPONSE),
        ("annotation_covariate", ColumnRole.COVARIATE),
    ):
        for column in getattr(args, attr, None) or []:
            previous = roles.get(column)
            if previous is not None and previous is not role:
                msg = f"Annotation column {column!r} was assigned multiple roles."
                raise ValueError(msg)
            roles[column] = role
    return roles


def _annotation_kinds(args: CliArgs) -> dict[str, ColumnKind]:
    kinds: dict[str, ColumnKind] = {}
    for attr, kind in (
        ("annotation_numeric", ColumnKind.NUMERIC),
        ("annotation_categorical", ColumnKind.CATEGORICAL),
    ):
        for column in getattr(args, attr, None) or []:
            previous = kinds.get(column)
            if previous is not None and previous is not kind:
                msg = f"Annotation column {column!r} was assigned multiple data kinds."
                raise ValueError(msg)
            kinds[column] = kind
    return kinds


def write_group_result(result: GroupAnalysisResult, output_dir: str | Path) -> Path:
    """Persist structured grouped-analysis artifacts."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    tables = {
        "response_catalog": result.response_catalog,
        "group_coverage": result.group_coverage,
        "grouped_numeric_summaries": result.numeric_summaries,
        "grouped_categorical_summaries": result.categorical_summaries,
        "grouped_numeric_comparisons": result.numeric_comparisons,
        "grouped_categorical_comparisons": result.categorical_comparisons,
        "annotation_coverage": result.annotation_coverage,
    }
    for name, table in tables.items():
        write_table(table, target / f"{name}.csv")
    write_json(result.provenance.to_dict(), target / "grouped_provenance.json")
    return target


def run_groups(args: CliArgs) -> int:
    """Run the ``ruddy analyze groups`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    sources = ()
    external_responses: tuple[str, ...] = ()
    if args.annotation_file:
        annotation_frame = read_table(args.annotation_file)
        aligned = align_annotation_source(
            dataset,
            annotation_frame,
            source_name=args.annotation_name,
            id_column=args.annotation_id_column,
            mode=args.annotation_alignment,
            role_overrides=_annotation_roles(args),
            kind_overrides=_annotation_kinds(args),
        )
        sources = (aligned,)
        external_responses = tuple(args.annotation_response or ())

    responses = tuple(config.responses) + external_responses
    groups = tuple(config.groups) if config.groups else None
    result = analyze_grouped_responses(
        dataset,
        annotations=sources,
        responses=responses or None,
        groups=groups,
        comparison_tests=config.comparison_tests,
        p_adjust=config.p_adjust,
        min_group_n=config.min_group_n,
        max_group_levels=config.max_group_levels,
        max_category_levels=config.max_category_levels,
        pairwise=config.pairwise,
    )
    if args.output_dir:
        print(write_group_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "responses": result.response_catalog.height,
                    "groups": result.group_coverage.height,
                    "numeric_summary_rows": result.numeric_summaries.height,
                    "categorical_summary_rows": result.categorical_summaries.height,
                    "numeric_comparisons": result.numeric_comparisons.height,
                    "categorical_comparisons": result.categorical_comparisons.height,
                    "annotation_sources": result.annotation_coverage.height,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0
