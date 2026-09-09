"""Factorial ANOVA/ANCOVA analysis with explicit designs and diagnostics."""

from ruddy.factorial.design import FactorialDesign, FactorialTerm, build_factorial_design
from ruddy.factorial.diagnostics import build_factorial_cells, model_diagnostics
from ruddy.factorial.effects import factorial_effect_sizes
from ruddy.factorial.models import FactorialResult, analyze_factorial

__all__ = [
    "FactorialDesign",
    "FactorialResult",
    "FactorialTerm",
    "analyze_factorial",
    "build_factorial_cells",
    "build_factorial_design",
    "factorial_effect_sizes",
    "model_diagnostics",
]
