"""CLI helpers for descriptive profiling and univariate analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ruddy.analysis import AnalysisConfig
from ruddy.core import ColumnKind, ColumnRole
from ruddy.data import TabularDataset
from ruddy.profiling import ProfilingResult, profile_dataset
from ruddy.univariate import UnivariateResult, analyze_univariate


def _build_role_overrides(args) -> dict[str, ColumnRole]:
    overrides: dict[str, ColumnRole] = {}
    for attr, role in (
        ("target", ColumnRole.TARGET),
        ("factor", ColumnRole.FACTOR),
        ("covariate", ColumnRole.COVARIATE),
        ("annotation", ColumnRole.ANNOTATION),
        ("exclude", ColumnRole.EXCLUDED),
    ):
        values = getattr(args, attr, None)
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        for column in values:
            if column in overrides and overrides[column] is not role:
                raise ValueError(
                    f"Column {column!r} was assigned multiple CLI roles: "
                    f"{overrides[column].value!r} and {role.value!r}."
                )
            overrides[column] = role
    return overrides


def _build_kind_overrides(args) -> dict[str, ColumnKind]:
    overrides: dict[str, ColumnKind] = {}
    for attr, kind in (
        ("numeric", ColumnKind.NUMERIC),
        ("categorical", ColumnKind.CATEGORICAL),
    ):
        for column in getattr(args, attr, None) or []:
            if column in overrides and overrides[column] is not kind:
                raise ValueError(
                    f"Column {column!r} was assigned multiple CLI kinds."
                )
            overrides[column] = kind
    return overrides


def build_config(args) -> AnalysisConfig:
    """Translate argparse options into an AnalysisConfig."""

    return AnalysisConfig(
        id_column=args.id_column,
        role_overrides=_build_role_overrides(args),
        kind_overrides=_build_kind_overrides(args),
        min_numeric_n=args.min_numeric_n,
        max_category_levels=args.max_category_levels,
        max_missingness_patterns=args.max_missingness_patterns,
        max_pairwise_columns=args.max_pairwise_columns,
    )


def load_dataset(path: str | Path, config: AnalysisConfig) -> TabularDataset:
    """Load a CSV/TSV table into Ruddy without scientific transformation."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(source, sep="\t")
    else:
        raise ValueError("Ruddy CLI currently accepts .csv, .tsv, or .txt tables.")
    return TabularDataset(frame, **config.dataset_kwargs())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tables(output_dir: Path, tables: Iterable[tuple[str, pd.DataFrame]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables:
        table.to_csv(output_dir / f"{name}.csv", index=False)


def write_profiling_result(result: ProfilingResult, output_dir: str | Path) -> Path:
    """Persist structured profiling outputs as JSON/CSV artifacts."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "overview.json", result.overview)
    _write_json(target / "provenance.json", result.provenance.to_dict())
    _write_tables(
        target,
        (
            ("columns", result.columns),
            ("missingness", result.missingness),
            ("pairwise_completeness", result.pairwise_completeness),
            ("missingness_patterns", result.missingness_patterns),
        ),
    )
    return target


def write_univariate_result(result: UnivariateResult, output_dir: str | Path) -> Path:
    """Persist structured univariate outputs as JSON/CSV artifacts."""

    target = write_profiling_result(result.profiling, output_dir)
    _write_json(target / "univariate_provenance.json", result.provenance.to_dict())
    _write_tables(
        target,
        (
            ("numeric_statistics", result.numeric_statistics),
            ("categorical_statistics", result.categorical_statistics),
            ("categorical_frequencies", result.categorical_frequencies),
            ("datetime_statistics", result.datetime_statistics),
        ),
    )
    return target


def run_profile(args) -> int:
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = profile_dataset(dataset, **config.profiling_kwargs())
    if args.output_dir:
        path = write_profiling_result(result, args.output_dir)
        print(path)
    else:
        print(json.dumps(result.overview, indent=2, sort_keys=True))
    return 0


def run_univariate(args) -> int:
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_univariate(dataset, **config.univariate_kwargs())
    if args.output_dir:
        path = write_univariate_result(result, args.output_dir)
        print(path)
    else:
        payload = {
            "overview": result.profiling.overview,
            "numeric_variables": int(len(result.numeric_statistics)),
            "categorical_variables": int(len(result.categorical_statistics)),
            "datetime_variables": int(len(result.datetime_statistics)),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
