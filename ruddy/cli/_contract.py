"""Shared ecosystem CLI contract for output-writing commands: `--config` and the provenance sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from ruddy._version import __version__
from ruddy.cli._config_file import apply_config
from ruddy.cli._console import STATE
from ruddy.cli._provenance import write_provenance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruddy.cli._options import CliArgs

#: Options that take a filesystem path; they stay on the command line, never in ``--config``.
PATH_OPTIONS = frozenset(
    {
        "output_dir",
        "ids_file",
        "x_ids_file",
        "y_ids_file",
        "feature_input",
        "feature_ids_file",
        "comparison_feature_input",
        "comparison_feature_ids_file",
    }
)
_INPUT_ARGUMENTS = ("input", "input_x", "input_y")
_PACKAGES = ["ruddy", "numpy", "pandas", "polars", "scipy", "scikit-learn", "statsmodels"]


def config_exclude(command: Any) -> frozenset[str]:
    """Names, among ``command``'s parameters, that must stay on the command line."""
    return PATH_OPTIONS.intersection(p.name for p in command.params) | {"config"}


def _config_callback(ctx: typer.Context, value: Path | None) -> Path | None:
    apply_config(ctx, value, exclude=config_exclude(ctx.command))
    return value


CONFIG = typer.Option(
    None,
    "--config",
    is_eager=True,
    callback=_config_callback,
    help="YAML/JSON file of option values (keys are option names). Command-line flags win.",
)


def write_sidecar(args: CliArgs, *, command: str, created_at: str) -> None:
    """Write ``<output-dir>.provenance.json`` when the command was given ``--output-dir``."""
    values = vars(args)
    if not values.get("output_dir"):
        return
    paths = PATH_OPTIONS.intersection(values) - {"output_dir"}
    skip = paths | {"output_dir", "config", "command", *_INPUT_ARGUMENTS}
    given = [values[k] for k in (*_INPUT_ARGUMENTS, *sorted(paths)) if values.get(k)]
    write_provenance(
        Path(values["output_dir"]),
        tool="ruddy",
        version=__version__,
        command=command,
        created_at=created_at,
        parameters={k: v for k, v in values.items() if k not in skip},
        inputs=[Path(p) for p in given],
        packages=_PACKAGES,
        extensions={"executed_blocks": [name for name, _ in STATE.blocks]},
    )
