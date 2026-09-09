from __future__ import annotations

import ruddy


def test_factorial_public_api_is_exposed():
    assert callable(ruddy.analyze_factorial)
    assert callable(ruddy.build_factorial_design)
    assert ruddy.FactorialResult.__name__ == "FactorialResult"
    assert ruddy.FactorialDesign.__name__ == "FactorialDesign"
