"""Ruddy command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ruddy._version import __version__
from ruddy.cli.commands import run_profile, run_univariate


def _add_descriptive_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input .csv, .tsv, or .txt table.")
    parser.add_argument("--id-column", default=None, help="Observation identifier column.")
    parser.add_argument("--target", default=None, help="Target/response column.")
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
