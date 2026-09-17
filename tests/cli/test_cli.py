import pytest
import typer

from ruddy import __version__
from ruddy.cli.main import app, main


def _command_names(group: str | None = None) -> list[str]:
    command = typer.main.get_command(app)
    commands = command.commands  # pyrefly: ignore[missing-attribute]
    if group is not None:
        commands = commands[group].commands
    return sorted(commands)


def test_cli_exposes_the_command_groups() -> None:
    assert _command_names() == ["analyze", "inspect", "model", "pipeline", "project"]


def test_cli_empty_invocation() -> None:
    # ``no_args_is_help`` prints the help and exits 2, as in the sibling CLIs.
    assert main([]) == 2


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"ruddy {__version__}"
