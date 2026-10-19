"""Ruddy command-line entry point."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

import ruddy.cli.analyze as analyze_group
import ruddy.cli.inspect as inspect_group
import ruddy.cli.model as model_group
import ruddy.cli.pipeline as pipeline_command
import ruddy.cli.project as project_group
from ruddy.cli._config_file import ConfigIssue, check_config, read_config, report_issues
from ruddy.cli._console import STATE, configure, fail
from ruddy.cli._contract import config_exclude
from ruddy.cli._options import CONTEXT_SETTINGS, bundle, version_callback
from ruddy.cli.commands.descriptive import build_config
from ruddy.core.exceptions import ResultContractError, RuddyError

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

app.command("pipeline", context_settings=CONTEXT_SETTINGS)(pipeline_command.pipeline)
app.add_typer(inspect_group.app, name="inspect")
app.add_typer(analyze_group.app, name="analyze")
app.add_typer(model_group.app, name="model")
app.add_typer(project_group.app, name="project")


VALIDATE_CONFIG = typer.Argument(..., metavar="CONFIG", help="Config file (YAML or JSON).")
VALIDATE_JSON = typer.Option(False, "--json", help="Print the issues as a JSON array on stdout.")
VALIDATE_COMMAND = typer.Option(
    "pipeline", "--command", help='Command whose options the config targets, e.g. "project pca".'
)


def _find_command(name: str) -> Any:
    command = typer.main.get_command(app)
    for part in name.split():
        command = getattr(command, "commands", {}).get(part)  # duck-typed: Typer vendors Click
        if command is None:
            msg = f"unknown command {name!r}"
            raise typer.BadParameter(msg, param_hint="'--command'")
    return command


@app.command("validate", context_settings=CONTEXT_SETTINGS)
def validate(
    config: Path = VALIDATE_CONFIG,
    as_json: bool = VALIDATE_JSON,
    command_name: str = VALIDATE_COMMAND,
) -> None:
    """Validate a ``--config`` file for a command (default ``pipeline``) without reading any data."""
    command = _find_command(command_name)
    raw, issues = read_config(config)
    issues = issues or check_config(command, raw, exclude=config_exclude(command))
    if not issues:
        # Parse with the config as defaults, then run the same semantic checks the command would.
        n_arguments = sum(getattr(p, "param_type_name", None) == "argument" for p in command.params)
        ctx = command.make_context(command_name, ["-"] * n_arguments, default_map=raw)
        if "min_numeric_n" in ctx.params:  # table commands share the AnalysisConfig checks
            try:
                build_config(bundle(ctx.params))
            except typer.BadParameter as exc:
                issues = [ConfigIssue("", exc.format_message())]
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
        # An invalid result is a bug, not bad input, so it exits 4 like any unexpected error.
        if isinstance(exc, RuddyError | OSError | ValueError) and not isinstance(exc, ResultContractError):
            fail(exc)
            return 3
        traceback.print_exc()
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
