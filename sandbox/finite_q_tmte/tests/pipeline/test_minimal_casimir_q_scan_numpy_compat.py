from __future__ import annotations

import pytest

from sandbox.finite_q_tmte.tmte.pipeline.minimal_casimir_q_scan import _trapezoid_q_integral


def test_q_scan_trapezoid_helper_does_not_require_numpy_trapz():
    rows = [
        {"q_magnitude": 0.0, "value": 0.0},
        {"q_magnitude": 1.0, "value": 2.0},
        {"q_magnitude": 2.0, "value": 2.0},
    ]
    assert _trapezoid_q_integral(rows, "value") == pytest.approx(3.0)
