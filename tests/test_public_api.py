import importlib

import pytest

import ruddy


@pytest.mark.parametrize("name", ruddy.__all__)
def test_public_name_resolves(name: str) -> None:
    assert hasattr(ruddy, name)


@pytest.mark.parametrize(
    "module",
    [
        "analysis",
        "anomaly",
        "bayesian",
        "bivariate",
        "cli",
        "compositional",
        "core",
        "data",
        "factorial",
        "multivariate",
        "profiling",
        "projections",
        "representation",
        "results",
        "statistics",
        "univariate",
    ],
)
def test_subpackage_imports(module: str) -> None:
    importlib.import_module(f"ruddy.{module}")
