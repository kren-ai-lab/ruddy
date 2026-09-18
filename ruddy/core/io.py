"""Centralised, extension-dispatched I/O for Ruddy tables and feature matrices.

Every read/write of an on-disk dataset goes through this module so that the
pandas backend stays replaceable from a single place. Nothing here performs a
scientific transformation: files are read as-is and written as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy import sparse

from ruddy.core.exceptions import RuddyIOError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

TAB_SEPARATED = frozenset({".tsv", ".txt"})
TABLE_EXTENSIONS = frozenset({".csv", ".parquet"}) | TAB_SEPARATED
MATRIX_EXTENSIONS = frozenset({".npy", ".npz"})
FEATURE_EXTENSIONS = TABLE_EXTENSIONS | MATRIX_EXTENSIONS


def _existing(path: str | Path, *, kind: str) -> Path:
    source = Path(path)
    if not source.exists():
        msg = f"{kind} not found: {source}"
        raise RuddyIOError(msg)
    return source


def _unsupported(suffix: str, supported: Iterable[str]) -> RuddyIOError:
    return RuddyIOError(f"Unsupported file extension {suffix!r}. Supported: {sorted(supported)}.")


def read_table(path: str | Path) -> pl.DataFrame:
    """Read a tabular file into a Polars DataFrame, dispatching on its extension.

    Supported extensions are defined by :data:`TABLE_EXTENSIONS` (``.csv``,
    ``.parquet``, ``.tsv``, ``.txt``).

    When reading a Parquet file written by pandas with an index column, that
    index column is retained as a regular data column because Polars frames do
    not use row indices.

    Raises:
        RuddyIOError: If the file is missing, has an unsupported extension, or
            cannot be parsed.

    """
    source = _existing(path, kind="Table")
    suffix = source.suffix.lower()
    if suffix not in TABLE_EXTENSIONS:
        raise _unsupported(suffix, TABLE_EXTENSIONS)
    try:
        if suffix == ".parquet":
            return pl.read_parquet(source)
        separator = "\t" if suffix in TAB_SEPARATED else ","
        return pl.read_csv(source, separator=separator, infer_schema_length=None)
    except RuddyIOError:
        raise
    except Exception as exc:
        msg = f"Could not read table {source}: {exc}"
        raise RuddyIOError(msg) from exc


def write_table(
    frame: pl.DataFrame,
    path: str | Path,
) -> Path:
    """Write a Polars DataFrame, dispatching on extension and creating parent dirs.

    Raises:
        RuddyIOError: If the extension is unsupported or the write fails.

    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in TABLE_EXTENSIONS:
        raise _unsupported(suffix, TABLE_EXTENSIONS)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if suffix == ".parquet":
            frame.write_parquet(target)
        else:
            frame.write_csv(target, separator="\t" if suffix in TAB_SEPARATED else ",")
    except Exception as exc:
        msg = f"Could not write table to {target}: {exc}"
        raise RuddyIOError(msg) from exc
    return target


def read_matrix(path: str | Path) -> np.ndarray | sparse.spmatrix:
    """Read a dense ``.npy`` or SciPy-sparse ``.npz`` feature matrix.

    Raises:
        RuddyIOError: If the file is missing, has an unsupported extension, or
            is not a readable array/sparse matrix.

    """
    source = _existing(path, kind="Feature matrix")
    suffix = source.suffix.lower()
    if suffix == ".npy":
        try:
            return np.load(source)
        except Exception as exc:
            msg = f"Could not read dense matrix {source}: {exc}"
            raise RuddyIOError(msg) from exc
    if suffix == ".npz":
        try:
            return sparse.load_npz(source)
        except Exception as exc:
            msg = (
                f"Could not read {source}: Ruddy expects .npz feature inputs to be "
                f"SciPy sparse matrices ({exc})."
            )
            raise RuddyIOError(msg) from exc
    raise _unsupported(suffix, MATRIX_EXTENSIONS)


def read_ids(path: str | Path) -> list[str]:
    """Read one observation identifier per line, ignoring blank lines.

    Raises:
        RuddyIOError: If the file is missing or unreadable.

    """
    source = _existing(path, kind="Observation ID file")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        msg = f"Could not read observation ID file {source}: {exc}"
        raise RuddyIOError(msg) from exc
    return [value for value in (line.strip() for line in lines) if value]


def write_json(payload: object, path: str | Path) -> Path:
    """Write a JSON artifact with Ruddy's canonical formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        msg = f"Could not write JSON to {target}: {exc}"
        raise RuddyIOError(msg) from exc
    return target


__all__ = [
    "FEATURE_EXTENSIONS",
    "MATRIX_EXTENSIONS",
    "TABLE_EXTENSIONS",
    "read_ids",
    "read_matrix",
    "read_table",
    "write_json",
    "write_table",
]
