"""CLI skeleton tests."""

import pytest
import typer

from ruddy.cli.main import app, main


def _command_names(group: str | None = None) -> list[str]:
    command = typer.main.get_command(app)
    if group is not None:
        command = command.commands[group]
    return sorted(command.commands)


def test_cli_exposes_the_command_groups() -> None:
    assert _command_names() == ["analyze", "inspect", "model", "project"]


def test_cli_empty_invocation() -> None:
    # ``no_args_is_help`` prints the help and exits 2, as in the sibling CLIs.
    assert main([]) == 2


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "ruddy 0.1.0.dev0"


def test_phase2_cli_exposes_profile_and_univariate() -> None:
    assert {"profile", "univariate"} <= set(_command_names("inspect"))
