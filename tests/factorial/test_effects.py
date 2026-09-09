from __future__ import annotations

import pytest

from ruddy.factorial import factorial_effect_sizes


def test_factorial_effect_sizes_match_closed_form():
    result = factorial_effect_sizes(
        effect_ss=30.0,
        effect_df=2.0,
        residual_ss=70.0,
        residual_df=35.0,
        corrected_total_ss=120.0,
    )
    mse = 2.0
    assert result["eta_squared"] == pytest.approx(30.0 / 120.0)
    assert result["partial_eta_squared"] == pytest.approx(30.0 / 100.0)
    assert result["omega_squared"] == pytest.approx((30.0 - 2.0 * mse) / (120.0 + mse))
    assert result["partial_omega_squared"] == pytest.approx((30.0 - 2.0 * mse) / (30.0 + 70.0 + mse))


def test_negative_omega_is_bounded_at_zero():
    result = factorial_effect_sizes(
        effect_ss=0.1,
        effect_df=2.0,
        residual_ss=100.0,
        residual_df=20.0,
        corrected_total_ss=100.1,
    )
    assert result["omega_squared"] == 0.0
    assert result["partial_omega_squared"] == 0.0


def test_degenerate_effect_size_inputs_return_none():
    result = factorial_effect_sizes(
        effect_ss=2.0,
        effect_df=1.0,
        residual_ss=2.0,
        residual_df=0.0,
        corrected_total_ss=4.0,
    )
    assert set(result.values()) == {None}
