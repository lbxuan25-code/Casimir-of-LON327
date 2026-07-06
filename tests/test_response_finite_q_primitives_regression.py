from dataclasses import fields

import numpy as np
import pytest

from lno327.models.lno327_four_orbital.bdg import bdg_hamiltonian
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.models.lno327_four_orbital.pairing import pairing_matrix
from lno327.response.config import KuboConfig
from lno327.response.finite_q import (
    BdGFiniteQResponseComponents,
    add_band_bubble,
    add_bubble,
    fermi_derivative,
    kubo_factor,
    thermal_expectation_bdg_from_hamiltonian,
    vertex_band,
)


def test_bdg_finite_q_response_components_fields_and_alias_are_stable():
    assert [field.name for field in fields(BdGFiniteQResponseComponents)] == [
        "bare_bubble",
        "direct",
        "bare_total",
        "phase_coupling_left",
        "phase_coupling_right",
        "phase_phase_bubble",
        "phase_phase_direct",
        "phase_phase_total",
        "minus_schur",
        "plus_schur",
        "collective_bubble",
        "collective_counterterm",
        "collective_total",
        "em_collective_left",
        "collective_em_right",
        "amplitude_phase_schur",
        "gauge_restored",
        "metadata",
    ]
    matrix = np.eye(2, dtype=complex)
    component = BdGFiniteQResponseComponents(
        bare_bubble=matrix,
        direct=matrix,
        bare_total=matrix,
        phase_coupling_left=np.ones((2,), dtype=complex),
        phase_coupling_right=np.ones((2,), dtype=complex),
        phase_phase_bubble=1 + 1j,
        phase_phase_direct=2 + 0j,
        phase_phase_total=3 + 1j,
        minus_schur=matrix,
        plus_schur=matrix,
        collective_bubble=matrix,
        collective_counterterm=matrix,
        collective_total=matrix,
        em_collective_left=matrix,
        collective_em_right=matrix,
        amplitude_phase_schur=matrix,
        gauge_restored=matrix,
        metadata={"ok": True},
    )

    assert component.phase_phase == component.phase_phase_total


@pytest.mark.parametrize("temperature_eV", [0.0, 0.02])
def test_fermi_derivative_is_nonpositive_and_finite(temperature_eV):
    value = fermi_derivative(0.03, 0.01, temperature_eV, 1e-4)
    assert np.isfinite(value)
    assert value <= 0.0


def test_kubo_factor_dynamic_and_static_cases():
    args = (-0.2, 0.3, 0.9, 0.1, 0.08)

    assert kubo_factor(*args) == (0.9 - 0.1) / (1j * 0.08 + (-0.2 - 0.3))
    static_value = kubo_factor(
        0.01,
        0.01 + 1e-10,
        0.4,
        0.4,
        0.0,
        static_limit=True,
        fermi_level_eV=0.0,
        temperature_eV=0.02,
        eta_eV=1e-4,
    )
    assert np.isfinite(static_value)

    with pytest.raises(ValueError, match="temperature_eV is required"):
        kubo_factor(0.01, 0.01, 0.4, 0.4, 0.0, static_limit=True, eta_eV=1e-4)


def test_vertex_band_returns_forward_vertex_in_minus_plus_storage():
    rng = np.random.default_rng(1234)
    states_minus = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    states_plus = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    vertex = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))

    np.testing.assert_allclose(
        vertex_band(states_minus, vertex, states_plus),
        (states_plus.conjugate().T @ vertex @ states_minus).T,
    )


def _bubble_inputs():
    rng = np.random.default_rng(5678)
    energies_minus = np.array([-0.2, 0.05, 0.3])
    energies_plus = np.array([-0.1, 0.07, 0.4])
    occupations_minus = np.array([0.9, 0.45, 0.1])
    occupations_plus = np.array([0.85, 0.4, 0.05])
    states_minus = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    states_plus = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    left_vertices = tuple(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)) for _ in range(2))
    right_vertices = tuple(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)) for _ in range(2))
    return (
        left_vertices,
        right_vertices,
        energies_minus,
        states_minus,
        occupations_minus,
        energies_plus,
        states_plus,
        occupations_plus,
    )


@pytest.mark.parametrize("use_config", [False, True])
def test_add_bubble_dynamic_cases_are_finite(use_config):
    inputs = _bubble_inputs()
    new_accumulator = np.zeros((2, 2), dtype=complex)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)

    add_bubble(
        new_accumulator,
        *inputs,
        0.08,
        0.37,
        new_config if use_config else None,
    )

    assert new_accumulator.shape == (2, 2)
    assert np.all(np.isfinite(new_accumulator))


def test_add_bubble_static_degenerate_branch_is_finite():
    inputs = list(_bubble_inputs())
    inputs[2] = np.array([0.01, 0.02, 0.03])
    inputs[5] = np.array([0.01 + 1e-10, 0.02 + 1e-10, 0.03 + 1e-10])
    inputs[4] = np.array([0.4, 0.4, 0.4])
    inputs[7] = np.array([0.4, 0.4, 0.4])
    new_accumulator = np.zeros((2, 2), dtype=complex)
    new_config = KuboConfig(omega_eV=0.0, temperature_eV=0.02, eta_eV=1e-4)

    add_bubble(new_accumulator, *inputs, 0.0, 0.37, new_config, True)

    assert new_accumulator.shape == (2, 2)
    assert np.all(np.isfinite(new_accumulator))


def _random_hermitian(rng, dim):
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return 0.5 * (matrix + matrix.conjugate().T)


def _manual_forward_band_bubble(
    left_band_minus_plus,
    right_band_minus_plus,
    energies_minus,
    occupations_minus,
    energies_plus,
    occupations_plus,
    omega_eV,
    weight,
):
    value = 0.0 + 0.0j
    for m, energy_minus in enumerate(energies_minus):
        for n, energy_plus in enumerate(energies_plus):
            raw_factor = float(occupations_minus[m] - occupations_plus[n]) / (
                1j * float(omega_eV) + float(energy_minus - energy_plus)
            )
            value += (
                0.5
                * float(weight)
                * raw_factor
                * left_band_minus_plus[m, n]
                * np.conjugate(right_band_minus_plus[m, n])
            )
    return value


def _forward_bubble_fixture(seed):
    rng = np.random.default_rng(seed)
    dim = 4
    omega = 0.13
    weight = 0.37
    h_minus = _random_hermitian(rng, dim)
    h_plus = _random_hermitian(rng, dim)
    energies_minus, states_minus = np.linalg.eigh(h_minus)
    energies_plus, states_plus = np.linalg.eigh(h_plus)
    occupations_minus = np.array([0.9, 0.7, 0.25, 0.1])
    occupations_plus = np.array([0.85, 0.55, 0.2, 0.05])
    left_vertex = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    right_vertex = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    left_band = (states_plus.conjugate().T @ left_vertex @ states_minus).T
    right_band = (states_plus.conjugate().T @ right_vertex @ states_minus).T
    return {
        "energies_minus": energies_minus,
        "states_minus": states_minus,
        "occupations_minus": occupations_minus,
        "energies_plus": energies_plus,
        "states_plus": states_plus,
        "occupations_plus": occupations_plus,
        "left_vertex": left_vertex,
        "right_vertex": right_vertex,
        "left_band": left_band,
        "right_band": right_band,
        "omega": omega,
        "weight": weight,
    }


def test_add_band_bubble_matches_forward_trace_convention():
    fixture = _forward_bubble_fixture(20260707)
    expected = _manual_forward_band_bubble(
        fixture["left_band"],
        fixture["right_band"],
        fixture["energies_minus"],
        fixture["occupations_minus"],
        fixture["energies_plus"],
        fixture["occupations_plus"],
        fixture["omega"],
        fixture["weight"],
    )

    got = np.zeros((1, 1), dtype=complex)
    add_band_bubble(
        got,
        (fixture["left_band"],),
        (fixture["right_band"],),
        fixture["energies_minus"],
        fixture["occupations_minus"],
        fixture["energies_plus"],
        fixture["occupations_plus"],
        fixture["omega"],
        fixture["weight"],
    )

    assert got[0, 0] == pytest.approx(expected)


def test_add_bubble_matrix_vertices_match_forward_trace_convention():
    fixture = _forward_bubble_fixture(20260708)
    expected = _manual_forward_band_bubble(
        fixture["left_band"],
        fixture["right_band"],
        fixture["energies_minus"],
        fixture["occupations_minus"],
        fixture["energies_plus"],
        fixture["occupations_plus"],
        fixture["omega"],
        fixture["weight"],
    )

    got = np.zeros((1, 1), dtype=complex)
    add_bubble(
        got,
        (fixture["left_vertex"],),
        (fixture["right_vertex"],),
        fixture["energies_minus"],
        fixture["states_minus"],
        fixture["occupations_minus"],
        fixture["energies_plus"],
        fixture["states_plus"],
        fixture["occupations_plus"],
        fixture["omega"],
        fixture["weight"],
    )

    assert got[0, 0] == pytest.approx(expected)


def test_thermal_expectation_from_hamiltonian_is_finite_and_complex():
    kx, ky = 0.21, -0.34
    amp = PairingAmplitudes(delta0_eV=0.04)
    delta = pairing_matrix("spm", kx, ky, amp)
    hamiltonian = bdg_hamiltonian(kx, ky, delta)
    vertex = np.eye(8, dtype=complex)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)

    new = thermal_expectation_bdg_from_hamiltonian(hamiltonian, vertex, new_config)

    assert isinstance(new, complex)
    assert np.isfinite(new)
