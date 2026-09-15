"""Shared Rich console and end-of-command reporting for the Ruddy CLI.

Everything rendered here goes to **stderr**: several commands print machine
readable JSON to stdout and that stream must stay uncontaminated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from ruddy.data import FeatureMatrix, TabularDataset

console = Console(stderr=True, highlight=False)

OK_MARK = "✓"
WARN_MARK = "⚠"


@dataclass
class _RunState:
    """Facts collected while a single CLI invocation runs."""

    quiet: bool = False
    debug: bool = False
    inputs: list[tuple[str, str]] = field(default_factory=list)
    blocks: list[tuple[str, str]] = field(default_factory=list)
    output: Path | None = None


STATE = _RunState()


def configure(*, quiet: bool, debug: bool) -> None:
    """Reset per-invocation state and apply the global flags."""
    global STATE  # noqa: PLW0603
    STATE = _RunState(quiet=quiet, debug=debug)


def record_dataset(dataset: TabularDataset) -> None:
    """Record tabular dataset shape for the end-of-command summary."""
    STATE.inputs.append(
        (
            "dataset",
            f"{dataset.n_observations} observations, {dataset.n_columns} columns "
            f"({len(dataset.columns_with_kind('numeric'))} numeric, "
            f"{len(dataset.columns_with_kind('categorical'))} categorical)",
        )
    )


def record_features(features: FeatureMatrix) -> None:
    """Record feature-matrix shape for the end-of-command summary."""
    STATE.inputs.append(
        (
            "features",
            f"{features.n_observations} observations, {features.n_features} features",
        )
    )


def record_blocks(blocks: Iterable[tuple[str, bool]]) -> None:
    """Record executed analysis blocks as ``(name, ok)`` pairs."""
    STATE.blocks = [(name, OK_MARK if ok else WARN_MARK) for name, ok in blocks]


def record_output(path: str | Path | None) -> None:
    """Record where the command's artifacts were written."""
    STATE.output = None if path is None else Path(path)


def render(command: str) -> None:
    """Render the run summary to stderr unless ``--quiet`` was requested."""
    if STATE.quiet:
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    for label, value in STATE.inputs:
        table.add_row(label, value)
    if STATE.blocks:
        table.add_row(
            "blocks",
            "  ".join(f"{mark} {name}" for name, mark in STATE.blocks),
        )
    table.add_row("results", str(STATE.output) if STATE.output else "stdout")
    console.print(Panel(table, title=f"ruddy {command}", title_align="left"))


def fail(exc: BaseException) -> None:
    """Report a user-facing error on stderr, or re-raise it under ``--debug``."""
    if STATE.debug:
        raise exc
    console.print(f"[bold red]Error[/bold red] ({type(exc).__name__}): {exc}")


__all__ = [
    "STATE",
    "configure",
    "console",
    "fail",
    "record_blocks",
    "record_dataset",
    "record_features",
    "record_output",
    "render",
]
