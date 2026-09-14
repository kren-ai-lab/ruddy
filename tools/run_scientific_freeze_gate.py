#!/usr/bin/env python3
"""Run the Ruddy scientific-freeze validation gates.

This script intentionally performs only scientific/runtime checks. Product hardening,
formatting, documentation, notebooks, and release engineering belong to later phases.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(root: Path, *args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=root, check=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    _run(root, sys.executable, "-m", "compileall", "-q", "ruddy", "tests")
    _run(root, sys.executable, "-m", "pytest", "-q", "tests/phase10")
    _run(root, sys.executable, "-m", "pytest", "-q")
    print("Ruddy scientific freeze gate PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
