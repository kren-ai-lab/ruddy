"""Effect-size calculations for factorial ANOVA and ANCOVA."""

from __future__ import annotations

import math


def _bounded(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return float(min(1.0, max(0.0, value)))


def factorial_effect_sizes(
    *,
    effect_ss: float,
    effect_df: float,
    residual_ss: float,
    residual_df: float,
    corrected_total_ss: float,
) -> dict[str, float | None]:
    """Compute common effect sizes from one factorial ANOVA/ANCOVA term.

    ``eta_squared`` is referenced to the corrected total sum of squares. In Type II
    or Type III analyses, term sums of squares are adjusted and do not necessarily
    partition the corrected total; effect sizes therefore should not be interpreted
    as additive shares across terms.
    """

    values = (effect_ss, effect_df, residual_ss, residual_df, corrected_total_ss)
    if not all(math.isfinite(float(value)) for value in values):
        return {
            "eta_squared": None,
            "partial_eta_squared": None,
            "omega_squared": None,
            "partial_omega_squared": None,
        }
    if effect_df <= 0.0 or residual_df <= 0.0 or residual_ss < 0.0:
        return {
            "eta_squared": None,
            "partial_eta_squared": None,
            "omega_squared": None,
            "partial_omega_squared": None,
        }

    mse = residual_ss / residual_df
    eta = effect_ss / corrected_total_ss if corrected_total_ss > 0.0 else math.nan
    partial_eta_den = effect_ss + residual_ss
    partial_eta = effect_ss / partial_eta_den if partial_eta_den > 0.0 else math.nan

    numerator = effect_ss - effect_df * mse
    omega_den = corrected_total_ss + mse
    omega = numerator / omega_den if omega_den > 0.0 else math.nan
    partial_omega_den = effect_ss + residual_ss + mse
    partial_omega = numerator / partial_omega_den if partial_omega_den > 0.0 else math.nan

    return {
        "eta_squared": _bounded(eta),
        "partial_eta_squared": _bounded(partial_eta),
        "omega_squared": _bounded(omega),
        "partial_omega_squared": _bounded(partial_omega),
    }


__all__ = ["factorial_effect_sizes"]
