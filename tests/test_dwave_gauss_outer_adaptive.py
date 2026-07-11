from __future__ import annotations

import numpy as np
import pytest

from validation.lib.gauss_outer_adaptive import gauss_outer_adaptive_integral
from validation.lib.iterated_adaptive import EvaluationBudgetExceeded, IteratedAdaptiveOptions


def test_gauss_outer_adaptive_matches_analytic_vector_in_both_orders():
    def integrand(kx: float, ky: float) -> np.ndarray:
        return np.asarray([1.0, kx, ky, kx * ky, kx * kx + ky * ky])

    options = IteratedAdaptiveOptions(
        epsabs=1e-11,
        epsrel=1e-11,
        inner_limit=40,
        outer_limit=1,
        max_point_evaluations=50_000,
        quadrature="gk15",
        split_points=(0.0,),
    )
    expected = np.asarray(
        [4.0 * np.pi**2, 0.0, 0.0, 0.0, 8.0 * np.pi**4 / 3.0]
    )
    xy = gauss_outer_adaptive_integral(
        integrand, order="xy", outer_order=8, options=options
    )
    yx = gauss_outer_adaptive_integral(
        integrand, order="yx", outer_order=8, options=options
    )
    assert xy.success
    assert yx.success
    assert xy.outer_evaluations == 8
    assert yx.outer_evaluations == 8
    assert np.allclose(xy.value, expected, rtol=1e-11, atol=1e-11)
    assert np.allclose(yx.value, expected, rtol=1e-11, atol=1e-11)
    assert np.allclose(xy.value, yx.value, rtol=1e-12, atol=1e-12)
    assert "excludes outer discretization" in xy.message


def test_gauss_outer_adaptive_point_budget_is_fail_closed():
    options = IteratedAdaptiveOptions(
        max_point_evaluations=1,
        inner_limit=10,
        outer_limit=1,
    )
    with pytest.raises(EvaluationBudgetExceeded):
        gauss_outer_adaptive_integral(
            lambda kx, ky: np.asarray([kx + ky]),
            order="xy",
            outer_order=4,
            options=options,
        )


def test_gauss_outer_adaptive_validates_outer_order():
    with pytest.raises(ValueError):
        gauss_outer_adaptive_integral(
            lambda kx, ky: np.asarray([kx + ky]),
            order="xy",
            outer_order=0,
            options=IteratedAdaptiveOptions(),
        )
