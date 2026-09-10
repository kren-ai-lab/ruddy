"""Validate every Ruddy Phase-11 visualization notebook.

Default ``python`` mode executes all code cells in an isolated Python process per
notebook with a non-interactive Matplotlib backend. ``jupyter`` mode performs a
full notebook execution through nbconvert and can optionally persist outputs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nbformat


def _python_gate(notebook: Path, root: Path, timeout: int) -> int:
    document = nbformat.read(notebook, as_version=4)
    source = "\n\n# ---- notebook cell ----\n\n".join(
        cell.source for cell in document.cells if cell.cell_type == "code"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        handle.write(source)
        script = Path(handle.name)
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MPLBACKEND"] = "Agg"
    try:
        completed = subprocess.run(
            [sys.executable, str(script)], cwd=root, env=env, timeout=timeout,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"FAILED: {notebook.name}: Python execution timed out", file=sys.stderr)
        return 1
    finally:
        script.unlink(missing_ok=True)
    if completed.returncode:
        print(f"FAILED: {notebook.name}\n{completed.stderr}", file=sys.stderr)
    return completed.returncode


def _jupyter_gate(notebook: Path, root: Path, timeout: int, inplace: bool) -> int:
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    if inplace:
        command.append("--inplace")
    else:
        command.extend(["--output", notebook.stem + ".validated.ipynb"])
    command.append(str(notebook))
    try:
        completed = subprocess.run(command, cwd=root, env=env, timeout=timeout * 3, check=False)
    except subprocess.TimeoutExpired:
        print(f"FAILED: {notebook.name}: Jupyter execution timed out", file=sys.stderr)
        return 1
    if not inplace:
        generated = notebook.with_name(notebook.stem + ".validated.ipynb")
        generated.unlink(missing_ok=True)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=240, help="Execution timeout in seconds.")
    parser.add_argument("--mode", choices=("python", "jupyter"), default="python")
    parser.add_argument("--inplace", action="store_true", help="With --mode jupyter, persist freshly executed outputs.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "examples" / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    if not notebooks:
        print("No Phase-11 notebooks found.", file=sys.stderr)
        return 2

    for notebook in notebooks:
        print(f"[Ruddy notebooks] validating {notebook.name} ({args.mode})", flush=True)
        if args.mode == "python":
            code = _python_gate(notebook, root, args.timeout)
        else:
            code = _jupyter_gate(notebook, root, args.timeout, args.inplace)
        if code:
            return code

    print(f"Ruddy Phase-11 notebook gate PASSED ({len(notebooks)} notebooks, mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
