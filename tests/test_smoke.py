"""Skeleton smoke tests."""


def test_import_ruddy() -> None:
    import ruddy

    assert ruddy.__version__ == "0.1.0.dev0"


def test_scientific_namespaces_import() -> None:
    import ruddy.analysis
    import ruddy.bivariate
    import ruddy.cli
    import ruddy.core
    import ruddy.data
    import ruddy.factorial
    import ruddy.multivariate
    import ruddy.profiling
    import ruddy.projections
    import ruddy.results
    import ruddy.statistics
    import ruddy.univariate  # noqa: F401
