"""Public config validation, shared with ``ruddy validate-config``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from ruddy.cli._config_file import ConfigIssue, check_config, read_config

if TYPE_CHECKING:
    import os


def validate_config(
    config: Mapping[str, Any] | str | os.PathLike[str], command: str = "pipeline"
) -> list[ConfigIssue]:
    """Check a config (mapping, or path to a YAML/JSON file) against ``command``'s options.

    ``command`` is the CLI's ``COMMAND`` value, e.g. ``"project pca"``. Returns the issues found (empty when
    valid); problems with the file or the command are issues with path ``""``. Reads no data.
    """
    # Lazy: the CLI pulls in every command module.
    from ruddy.cli._contract import config_exclude  # noqa: PLC0415
    from ruddy.cli._options import bundle  # noqa: PLC0415
    from ruddy.cli.commands.descriptive import build_config  # noqa: PLC0415
    from ruddy.cli.main import _find_command  # noqa: PLC0415

    try:
        cmd = _find_command(command)
    except typer.BadParameter as exc:
        return [ConfigIssue("", exc.format_message())]
    if isinstance(config, Mapping):
        raw, issues = dict(config), []
    else:
        raw, issues = read_config(Path(config))
    issues = issues or check_config(cmd, raw, exclude=config_exclude(cmd))
    if not issues:
        # Parse with the config as defaults, then run the same semantic checks the command would.
        n_arguments = sum(getattr(p, "param_type_name", None) == "argument" for p in cmd.params)
        try:
            ctx = cmd.make_context(command, ["-"] * n_arguments, default_map=raw)
            if "min_numeric_n" in ctx.params:  # table commands share the AnalysisConfig checks
                build_config(bundle(ctx.params))
        except typer.BadParameter as exc:
            issues = [ConfigIssue("", exc.format_message())]
    return issues
