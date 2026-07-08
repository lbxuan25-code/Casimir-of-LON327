from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.primitive_response_closure_suite import (
    DEFAULT_CANDIDATE,
    SCHEMA_VERSION,
    best_phase_grid,
    cancellation_angle,
    fit_one_scale,
    fit_two_scales,
)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_primitive_response_closure_suite.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_primitive_response_closure_suite_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_and_default_candidate():
    assert SCHEMA_VERSION == "finite_q_tmte_primitive_response_closure_suite_v1"
    assert DEFAULT_CANDIDATE == "matrix_inferred_matsubara_i_asymmetric"


def test_fit_one_scale_complex_closes_exact_vector():
    target = np.asarray([1.0 + 2.0j, -0.5j], dtype=complex)
    basis = np.asarray([2.0 - 1.0j, 0.25 + 0.25j], dtype=complex)
    coeff = -0.3 + 0.7j
    fit = fit_one_scale(target + coeff * basis, basis, real_only=False)
    assert fit["residual_norm"] < 1e-12
    np.testing.assert_allclose(fit["coefficient"], -coeff, atol=1e-12)


def test_fit_one_scale_real_restricts_imaginary_part():
    target = np.asarray([2.0, 0.0], dtype=complex)
    basis = np.asarray([1.0, 0.0], dtype=complex)
    fit = fit_one_scale(target, basis, real_only=True)
    np.testing.assert_allclose(fit["coefficient"], -2.0 + 0.0j)
    assert fit["residual_norm"] < 1e-14


def test_fit_two_scales_complex_closes_exact_vector():
    fixed = np.asarray([1.0, 2.0j, -1.0], dtype=complex)
    first = np.asarray([1.0j, 0.5, 0.0], dtype=complex)
    second = np.asarray([0.0, -1.0j, 2.0], dtype=complex)
    c1 = 0.25 - 0.5j
    c2 = -1.5 + 0.1j
    fit = fit_two_scales(fixed + c1 * first + c2 * second, first, second, real_only=False)
    assert fit["residual_norm"] < 1e-12
    np.testing.assert_allclose(fit["coefficients"], [-c1, -c2], atol=1e-12)


def test_best_phase_grid_finds_cancellation_choice():
    fixed = np.asarray([1.0, 0.0], dtype=complex)
    first = np.asarray([1.0, 0.0], dtype=complex)
    second = np.asarray([0.0, 0.0], dtype=complex)
    best = best_phase_grid(fixed, first, second)
    assert best["first_coefficient"] == -1.0 + 0.0j
    assert best["residual_norm"] < 1e-14


def test_cancellation_angle_identifies_opposite_vectors():
    first = np.asarray([1.0, 2.0j], dtype=complex)
    second = -first
    report = cancellation_angle(first, second)
    assert report["sum_norm"] < 1e-14
    assert report["real_overlap"] == pytest.approx(-1.0)


def test_closure_suite_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "1",
                "--q",
                "0.02",
                "--nk",
                "0",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_closure_suite_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
                "--pairing",
                "dwave",
                "--matsubara-index",
                "-1",
                "--q",
                "0.02",
                "--nk",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )
