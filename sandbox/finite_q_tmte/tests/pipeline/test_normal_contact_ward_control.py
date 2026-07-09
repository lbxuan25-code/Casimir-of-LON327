from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sandbox.finite_q_tmte.tmte.pipeline.normal_contact_ward_control import (
    NORMAL_ORDER,
    SCHEMA_VERSION,
    contact_formula_payload,
    normal_peierls_vertex_ward_residual,
    scalar_projection,
    ward_residuals,
    ward_vectors,
)


class MockNormalSpec:
    def normal_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        return np.asarray([[2.0 * np.cos(kx) + 0.5 * np.cos(ky)]], dtype=complex)

    def peierls_hamiltonian_vector_vertex(self, kx: float, ky: float, qx: float, qy: float, direction: str) -> np.ndarray:
        if direction == "x":
            return np.asarray([[-2.0 * np.sin(kx) * np.sinc(qx / (2.0 * np.pi))]], dtype=complex)
        if direction == "y":
            return np.asarray([[-0.5 * np.sin(ky) * np.sinc(qy / (2.0 * np.pi))]], dtype=complex)
        raise ValueError(direction)

    def peierls_hamiltonian_contact_vertex(self, kx: float, ky: float, qx: float, qy: float, direction_i: str, direction_j: str) -> np.ndarray:
        if direction_i != direction_j:
            return np.zeros((1, 1), dtype=complex)
        if direction_i == "x":
            return np.asarray([[-2.0 * np.cos(kx) * np.sinc(qx / (2.0 * np.pi)) ** 2]], dtype=complex)
        if direction_i == "y":
            return np.asarray([[-0.5 * np.cos(ky) * np.sinc(qy / (2.0 * np.pi)) ** 2]], dtype=complex)
        raise ValueError(direction_i)


def _load_debug_script():
    path = Path("sandbox/finite_q_tmte/scripts/debug_normal_contact_ward_control.py").resolve()
    spec = importlib.util.spec_from_file_location("tmte_debug_normal_contact_ward_control_cli_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_schema_and_order():
    assert SCHEMA_VERSION == "finite_q_tmte_normal_contact_ward_control_v1"
    assert NORMAL_ORDER == ("density", "current_x", "current_y")


def test_ward_vectors_use_normal_left_right_signs():
    left, right = ward_vectors(0.2, np.asarray([0.3, -0.4]))
    np.testing.assert_allclose(left, [0.2j, 0.3, -0.4])
    np.testing.assert_allclose(right, [0.2j, -0.3, 0.4])


def test_ward_residuals_contract_matrix():
    matrix = np.eye(3, dtype=complex)
    left, right = ward_residuals(matrix, 0.2, np.asarray([0.3, -0.4]))
    np.testing.assert_allclose(left, [0.2j, 0.3, -0.4])
    np.testing.assert_allclose(right, [0.2j, -0.3, 0.4])


def test_scalar_projection_exact_parallel_vector():
    current = np.asarray([1.0, 2.0j, -1.0], dtype=complex)
    alpha = 0.75 + 0.1j
    report = scalar_projection(alpha * current, current)
    np.testing.assert_allclose(report["alpha_required_over_current"], alpha)
    assert report["residual_norm"] < 1e-12


def test_contact_formula_payload_alpha_is_one_when_contact_is_required():
    q = np.asarray([0.2, 0.0])
    xi = 0.1
    bubble = np.zeros((3, 3), dtype=complex)
    bubble[1, 1] = 2.0
    contact = np.zeros((3, 3), dtype=complex)
    # left required = -[i xi, q, 0] @ bubble has L component -q*2.
    # current = [i xi, q, 0] @ contact, so contact_LL=-2 gives current=-q*2.
    contact[1, 1] = -2.0
    payload = contact_formula_payload(bubble, contact, xi, q)
    left = payload["left_required_over_current_scalar_projection"]
    right = payload["right_required_over_current_scalar_projection"]
    np.testing.assert_allclose(left["alpha_required_over_current"], 1.0 + 0.0j)
    np.testing.assert_allclose(right["alpha_required_over_current"], 1.0 + 0.0j)
    assert left["residual_norm"] < 1e-14
    assert right["residual_norm"] < 1e-14


def test_normal_peierls_vertex_ward_residual_mock_spec_small_q():
    report = normal_peierls_vertex_ward_residual(MockNormalSpec(), 0.4, -0.2, 1e-4, 0.0)
    assert report["rel_error"] < 1e-8


def test_normal_contact_ward_control_cli_rejects_nonpositive_nk(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
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


def test_normal_contact_ward_control_cli_rejects_negative_matsubara_index(tmp_path):
    module = _load_debug_script()
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--model",
                "symmetry_bdg_2band",
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
