"""Ruddy command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

import ruddy.cli.analyze as analyze_group
import ruddy.cli.inspect as inspect_group
import ruddy.cli.model as model_group
import ruddy.cli.pipeline as pipeline_command
import ruddy.cli.project as project_group
from ruddy.cli._config_file import ConfigIssue, check_config, read_config, report_issues
from ruddy.cli._console import STATE, configure
from ruddy.cli._options import CONTEXT_SETTINGS, bundle, version_callback
from ruddy.cli.commands.descriptive import build_config
from ruddy.core.exceptions import RuddyError

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

pipeline_command_app = typer.Typer(add_completion=False)
pipeline_command_app.command("pipeline", context_settings=CONTEXT_SETTINGS)(pipeline_command.pipeline)
app.command("pipeline", context_settings=CONTEXT_SETTINGS)(pipeline_command.pipeline)
app.add_typer(inspect_group.app, name="inspect")
app.add_typer(analyze_group.app, name="analyze")
app.add_typer(model_group.app, name="model")
app.add_typer(project_group.app, name="project")


VALIDATE_CONFIG = typer.Argument(..., metavar="CONFIG", help="Pipeline config file (YAML or JSON).")
VALIDATE_JSON = typer.Option(False, "--json", help="Print the issues as a JSON array on stdout.")


@app.command("validate", context_settings=CONTEXT_SETTINGS)
def validate(config: Path = VALIDATE_CONFIG, as_json: bool = VALIDATE_JSON) -> None:
    """Validate a ``ruddy pipeline --config`` file without reading any data."""
    command = typer.main.get_command(pipeline_command_app)
    raw, issues = read_config(config)
    issues = issues or check_config(command, raw, exclude=pipeline_command.CONFIG_EXCLUDE)
    if not issues:
        # Parse with the config as defaults, then run the same semantic checks the pipeline would.
        ctx = command.make_context("pipeline", ["-"], default_map=raw)
        try:
            build_config(bundle(ctx.params))
        except (RuddyError, TypeError, ValueError) as exc:
            issues = [ConfigIssue("", str(exc))]
    raise typer.Exit(report_issues(issues, as_json=as_json))


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
