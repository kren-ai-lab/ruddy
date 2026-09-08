"""CLI skeleton tests."""

import pytest

from ruddy.cli.main import build_parser, main


def test_cli_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "ruddy"


def test_cli_empty_invocation() -> None:
    assert main([]) == 0


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "ruddy 0.1.0.dev0"


def test_phase2_cli_exposes_profile_and_univariate() -> None:
    parser = build_parser()
    assert parser.parse_args(["profile", "data.csv"]).command == "profile"
    assert parser.parse_args(["univariate", "data.csv"]).command == "univariate"
