"""Factorial ANOVA/ANCOVA analysis with explicit designs and diagnostics."""

from ruddy.factorial.design import FactorialDesign, FactorialTerm, build_factorial_design
from ruddy.factorial.diagnostics import build_factorial_cells, model_diagnostics
from ruddy.factorial.effects import factorial_effect_sizes
from ruddy.factorial.marginal_means import MarginalMeansResult, analyze_marginal_means
from ruddy.factorial.mixed_effects import MixedEffectsResult, analyze_mixed_effects
from ruddy.factorial.models import FactorialResult, analyze_factorial

__all__ = [
    "FactorialDesign",
    "FactorialResult",
    "FactorialTerm",
    "MarginalMeansResult",
    "MixedEffectsResult",
    "analyze_factorial",
    "analyze_marginal_means",
    "analyze_mixed_effects",
    "build_factorial_cells",
    "build_factorial_design",
    "factorial_effect_sizes",
    "model_diagnostics",
]
