"""Ruddy command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ruddy._version import __version__


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Ruddy CLI."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
