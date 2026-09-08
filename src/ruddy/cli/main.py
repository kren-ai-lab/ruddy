"""Ruddy command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ruddy._version import __version__
from ruddy.cli.commands import (
    run_bivariate,
    run_groups,
    run_outliers,
    run_profile,
    run_univariate,
)


def _add_descriptive_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input .csv, .tsv, or .txt table.")
    parser.add_argument("--id-column", default=None, help="Observation identifier column.")
    parser.add_argument("--response", action="append", default=[], help="Response column; repeatable.")
    parser.add_argument("--factor", action="append", default=[], help="Factor column; repeatable.")
    parser.add_argument("--covariate", action="append", default=[], help="Covariate column; repeatable.")
    parser.add_argument("--annotation", action="append", default=[], help="Annotation column; repeatable.")
    parser.add_argument("--exclude", action="append", default=[], help="Excluded column; repeatable.")
    parser.add_argument("--numeric", action="append", default=[], help="Force numeric kind without coercion; repeatable.")
    parser.add_argument("--categorical", action="append", default=[], help="Interpret as categorical; repeatable.")
    parser.add_argument("--min-numeric-n", type=int, default=3)
    parser.add_argument("--max-category-levels", type=int, default=50)
    parser.add_argument("--max-missingness-patterns", type=int, default=20)
    parser.add_argument("--max-pairwise-columns", type=int, default=200)
    parser.add_argument("--output-dir", default=None, help="Write structured CSV/JSON outputs to this directory.")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level Ruddy CLI parser."""

    parser = argparse.ArgumentParser(
        prog="ruddy",
        description=(
            "Domain-agnostic statistical exploratory data analysis for tabular "
            "datasets and feature representations."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    profile_parser = subparsers.add_parser("profile", help="Profile a tabular dataset.")
    _add_descriptive_options(profile_parser)
    profile_parser.set_defaults(handler=run_profile)

    univariate_parser = subparsers.add_parser(
        "univariate",
        help="Run profiling plus univariate descriptive statistics.",
    )
    _add_descriptive_options(univariate_parser)
    univariate_parser.set_defaults(handler=run_univariate)

    outliers_parser = subparsers.add_parser(
        "outliers",
        help="Run transparent univariate outlier and numeric quality diagnostics.",
    )
    _add_descriptive_options(outliers_parser)
    outliers_parser.add_argument(
        "--outlier-method",
        action="append",
        choices=("iqr", "robust_z"),
        default=None,
        help="Outlier rule; repeatable. Defaults to IQR and robust Z.",
    )
    outliers_parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    outliers_parser.add_argument("--robust-z-threshold", type=float, default=3.5)
    outliers_parser.add_argument(
        "--include-flags",
        action="store_true",
        help="Persist observation-level flags in addition to variable summaries.",
    )
    outliers_parser.set_defaults(handler=run_outliers)

    bivariate_parser = subparsers.add_parser(
        "bivariate",
        help="Run mixed-type bivariate associations and inference.",
    )
    _add_descriptive_options(bivariate_parser)
    bivariate_parser.add_argument(
        "--correlation",
        action="append",
        choices=("pearson", "spearman", "kendall"),
        default=None,
        help="Correlation method; repeatable. Defaults to all supported methods.",
    )
    bivariate_parser.add_argument(
        "--comparison-test",
        action="append",
        choices=(
            "welch_t",
            "mann_whitney",
            "welch_anova",
            "kruskal_wallis",
            "chi_square",
            "fisher_exact",
        ),
        default=None,
        help="Inferential test; repeatable. Defaults to all supported tests.",
    )
    bivariate_parser.add_argument("--min-group-n", type=int, default=3)
    bivariate_parser.add_argument("--min-correlation-pairs", type=int, default=3)
    bivariate_parser.add_argument("--max-group-levels", type=int, default=20)
    bivariate_parser.add_argument("--max-correlation-columns", type=int, default=100)
    bivariate_parser.add_argument(
        "--pairwise", action="store_true", help="Enable pairwise post-hoc tests for multi-group comparisons."
    )
    bivariate_parser.add_argument(
        "--p-adjust", choices=("none", "fdr_bh"), default="fdr_bh"
    )
    bivariate_parser.set_defaults(handler=run_bivariate)

    groups_parser = subparsers.add_parser(
        "groups",
        help="Analyze one or more statistical responses across grouping variables.",
    )
    _add_descriptive_options(groups_parser)
    groups_parser.add_argument(
        "--group", action="append", default=[],
        help="Grouping column; repeatable. Defaults to columns declared as factors.",
    )
    groups_parser.add_argument("--min-group-n", type=int, default=3)
    groups_parser.add_argument("--max-group-levels", type=int, default=20)
    groups_parser.add_argument(
        "--comparison-test",
        action="append",
        choices=(
            "welch_t", "mann_whitney", "welch_anova", "kruskal_wallis",
            "chi_square", "fisher_exact",
        ),
        default=None,
    )
    groups_parser.add_argument(
        "--p-adjust", choices=("none", "fdr_bh"), default="fdr_bh"
    )
    groups_parser.add_argument("--pairwise", action="store_true")
    groups_parser.add_argument("--annotation-file", default=None)
    groups_parser.add_argument("--annotation-name", default="annotations")
    groups_parser.add_argument("--annotation-id-column", default=None)
    groups_parser.add_argument(
        "--annotation-alignment", choices=("strict", "partial"), default="strict"
    )
    groups_parser.add_argument("--annotation-factor", action="append", default=[])
    groups_parser.add_argument("--annotation-response", action="append", default=[])
    groups_parser.add_argument("--annotation-covariate", action="append", default=[])
    groups_parser.add_argument("--annotation-numeric", action="append", default=[])
    groups_parser.add_argument("--annotation-categorical", action="append", default=[])
    groups_parser.set_defaults(handler=run_groups)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Ruddy CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        if argv is None or len(argv) == 0:
            parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
