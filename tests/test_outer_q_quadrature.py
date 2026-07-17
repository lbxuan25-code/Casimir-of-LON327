from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lno327.casimir.outer_quadrature import (
    build_outer_q_polar_grid,
    free_energy_per_area_from_logdet,
    integrate_outer_q,
    matsubara_prime_weights,
)
from lno327.constants import KB
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from validation.__main__ import resolve_command
from validation.commands.casimir.outer_q_quadrature_preflight import main


def _grid(*, radial: int = 8, angular: int = 16, offset: float = 0.5):
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    return build_outer_q_polar_grid(
        separation_m=20e-9,
        lattice_a_x_m=material.lattice_a_x_m,
        lattice_a_y_m=material.lattice_a_y_m,
        u_max=24.0,
        radial_order=radial,
        angular_order=angular,
        angular_offset_fraction=offset,
    )


def test_outer_q_grid_has_exact_disk_measure_and_no_q_zero_node() -> None:
    grid = _grid()
    assert grid.node_count == 8 * 16
    assert np.all(np.linalg.norm(grid.q_si_m_inv, axis=1) > 0.0)
    actual = integrate_outer_q(np.ones(grid.node_count), grid)
    assert np.isclose(actual, grid.disk_measure_m_inv2, rtol=2e-15, atol=0.0)
    assert grid.metadata["angular_symmetry_reduction"] is False
    assert grid.metadata["q_zero_node_present"] is False


def test_radial_gauss_rule_integrates_u_squared_with_fixed_measure() -> None:
    grid = _grid(radial=4)
    expected = grid.u_max**4 / (32.0 * np.pi * grid.separation_m**2)
    actual = integrate_outer_q(grid.u**2, grid)
    assert np.isclose(actual, expected, rtol=3e-15, atol=0.0)


def test_full_angle_rule_removes_fourfold_harmonic_and_is_cut_invariant() -> None:
    shifted = _grid(offset=0.5)
    unshifted = _grid(offset=0.0)
    shifted_value = integrate_outer_q(
        1.0 + 0.3 * np.cos(4.0 * shifted.phi_rad),
        shifted,
    )
    unshifted_value = integrate_outer_q(
        1.0 + 0.3 * np.cos(4.0 * unshifted.phi_rad),
        unshifted,
    )
    assert np.isclose(shifted_value, shifted.disk_measure_m_inv2, rtol=3e-15)
    assert np.isclose(unshifted_value, unshifted.disk_measure_m_inv2, rtol=3e-15)
    assert np.isclose(shifted_value, unshifted_value, rtol=3e-15)


def test_model_and_si_wavevectors_round_trip_componentwise() -> None:
    grid = _grid()
    recovered = np.column_stack(
        [
            grid.q_model[:, 0] / grid.lattice_a_x_m,
            grid.q_model[:, 1] / grid.lattice_a_y_m,
        ]
    )
    assert np.allclose(recovered, grid.q_si_m_inv, rtol=2e-16, atol=1e-9)


def test_matsubara_free_energy_applies_zero_half_weight_once() -> None:
    grid = _grid()
    assert np.array_equal(
        matsubara_prime_weights([0, 1, 3]),
        np.array([0.5, 1.0, 1.0]),
    )
    values = np.vstack(
        [
            np.full(grid.node_count, -0.2),
            np.full(grid.node_count, -0.1),
        ]
    )
    result = free_energy_per_area_from_logdet(
        values,
        matsubara_indices=[0, 1],
        temperature_K=10.0,
        grid=grid,
    )
    expected = KB * 10.0 * grid.disk_measure_m_inv2 * (0.5 * -0.2 + -0.1)
    assert np.isclose(result.total_J_m2, expected, rtol=2e-15, atol=0.0)
    assert result.metadata["partial_sum_only"] is True
    assert result.metadata["tail_included"] is False


def test_outer_q_preflight_route_and_json(tmp_path: Path) -> None:
    assert resolve_command("casimir", "outer-q-quadrature-preflight") == (
        "validation.commands.casimir.outer_q_quadrature_preflight"
    )
    output = tmp_path / "outer_q_preflight.json"
    main(
        [
            "--radial-order-low",
            "8",
            "--radial-order-high",
            "16",
            "--angular-order-low",
            "8",
            "--angular-order-high",
            "16",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "outer-q-quadrature-preflight-v1"
    assert payload["status"]["passed"] is True
    assert payload["status"]["outer_q_measure_contract_fixed"] is True
    assert payload["status"]["microscopic_outer_q_preflight_allowed"] is True
    assert payload["status"]["production_casimir_allowed"] is False
    assert payload["contract"]["angular_symmetry_reduction"] is False
    assert all(row["passed"] for row in payload["checks"].values())
