from pathlib import Path

import pytest

import ruddy


@pytest.mark.parametrize("name", ruddy.__all__)
def test_public_name_resolves(name: str) -> None:
    assert hasattr(ruddy, name)


def test_core_does_not_import_plotting_libraries() -> None:
    forbidden = ("import matplotlib", "from matplotlib", "import plotly", "from plotly")
    for path in Path(ruddy.__file__).parent.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
