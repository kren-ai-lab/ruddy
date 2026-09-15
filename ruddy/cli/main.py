"""Ruddy command-line entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

import ruddy.cli.analyze as analyze_group
import ruddy.cli.inspect as inspect_group
import ruddy.cli.model as model_group
import ruddy.cli.project as project_group
from ruddy.cli._console import STATE, configure
from ruddy.cli._options import CONTEXT_SETTINGS, version_callback

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

app = typer.Typer(
    name="ruddy",
    add_completion=False,
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Domain-agnostic statistical exploratory data analysis for tabular "
        "datasets and feature representations."
    ),
)

app.add_typer(inspect_group.app, name="inspect")
app.add_typer(analyze_group.app, name="analyze")
app.add_typer(model_group.app, name="model")
app.add_typer(project_group.app, name="project")

QUIET_OPTION = typer.Option(
    False,
    "--quiet",
    "-q",
    help="Suppress the Rich run summary on stderr.",
)
DEBUG_OPTION = typer.Option(
    False,
    "--debug",
    help="Re-raise errors with a full traceback instead of a clean message.",
)
VERSION_OPTION = typer.Option(
    None,
    "--version",
    "-v",
    callback=version_callback,
    is_eager=True,
    help="Show version and exit.",
)


@app.callback()
def root(
    quiet: bool = QUIET_OPTION,
    debug: bool = DEBUG_OPTION,
    _version: bool | None = VERSION_OPTION,
) -> None:
    """Ruddy CLI root."""
    configure(quiet=quiet, debug=debug)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI application and return a process exit code."""
    try:
        # Typer handles usage errors, aborts and interrupts itself, then calls
        # sys.exit(); we translate that into a return code for the entry point.
        typer.main.get_command(app).main(args=None if argv is None else list(argv), prog_name="ruddy")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        if STATE.debug:
            raise
        typer.echo(f"ERROR: {exc}", err=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
