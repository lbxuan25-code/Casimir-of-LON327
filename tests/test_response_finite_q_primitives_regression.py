from dataclasses import fields

import numpy as np
import pytest

from lno327.conductivity import KuboConfig as OldKuboConfig
from lno327.finite_q_primitives import (
    BdGFiniteQResponseComponents as OldComponents,
    add_bubble as old_add_bubble,
    fermi_derivative as old_fermi_derivative,
    kubo_factor as old_kubo_factor,
    thermal_expectation_bdg as old_thermal_expectation,
    vertex_band as old_vertex_band,
)
from lno327.models.lno327_four_orbital.bdg import bdg_hamiltonian
from lno327.models.lno327_four_orbital.pairing import pairing_matrix
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes
from lno327.response.config import KuboConfig
from lno327.response.finite_q import (
    BdGFiniteQResponseComponents,
    add_bubble,
    fermi_derivative,
    kubo_factor,
    thermal_expectation_bdg_from_hamiltonian,
    vertex_band,
)


def test_bdg_finite_q_response_components_fields_and_alias_match_legacy():
    assert [field.name for field in fields(BdGFiniteQResponseComponents)] == [
        field.name for field in fields(OldComponents)
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
def test_fermi_derivative_matches_legacy(temperature_eV):
    assert fermi_derivative(0.03, 0.01, temperature_eV, 1e-4) == old_fermi_derivative(
        0.03,
        0.01,
        temperature_eV,
        1e-4,
    )


def test_kubo_factor_matches_legacy_dynamic_and_static_cases():
    args = (-0.2, 0.3, 0.9, 0.1, 0.08)

    assert kubo_factor(*args) == old_kubo_factor(*args)
    assert kubo_factor(
        0.01,
        0.01 + 1e-10,
        0.4,
        0.4,
        0.0,
        static_limit=True,
        fermi_level_eV=0.0,
        temperature_eV=0.02,
        eta_eV=1e-4,
    ) == old_kubo_factor(
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

    with pytest.raises(ValueError, match="temperature_eV is required"):
        kubo_factor(0.01, 0.01, 0.4, 0.4, 0.0, static_limit=True, eta_eV=1e-4)
    with pytest.raises(ValueError, match="temperature_eV is required"):
        old_kubo_factor(0.01, 0.01, 0.4, 0.4, 0.0, static_limit=True, eta_eV=1e-4)


def test_vertex_band_matches_legacy():
    rng = np.random.default_rng(1234)
    states_minus = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    states_plus = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    vertex = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))

    np.testing.assert_allclose(
        vertex_band(states_minus, vertex, states_plus),
        old_vertex_band(states_minus, vertex, states_plus),
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
def test_add_bubble_matches_legacy_for_dynamic_cases(use_config):
    inputs = _bubble_inputs()
    old_accumulator = np.zeros((2, 2), dtype=complex)
    new_accumulator = np.zeros((2, 2), dtype=complex)
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)

    old_add_bubble(
        old_accumulator,
        *inputs,
        0.08,
        0.37,
        old_config if use_config else None,
    )
    add_bubble(
        new_accumulator,
        *inputs,
        0.08,
        0.37,
        new_config if use_config else None,
    )

    np.testing.assert_allclose(new_accumulator, old_accumulator)


def test_add_bubble_matches_legacy_for_static_degenerate_branch():
    inputs = list(_bubble_inputs())
    inputs[2] = np.array([0.01, 0.02, 0.03])
    inputs[5] = np.array([0.01 + 1e-10, 0.02 + 1e-10, 0.03 + 1e-10])
    inputs[4] = np.array([0.4, 0.4, 0.4])
    inputs[7] = np.array([0.4, 0.4, 0.4])
    old_accumulator = np.zeros((2, 2), dtype=complex)
    new_accumulator = np.zeros((2, 2), dtype=complex)
    old_config = OldKuboConfig(omega_eV=0.0, temperature_eV=0.02, eta_eV=1e-4)
    new_config = KuboConfig(omega_eV=0.0, temperature_eV=0.02, eta_eV=1e-4)

    old_add_bubble(old_accumulator, *inputs, 0.0, 0.37, old_config, True)
    add_bubble(new_accumulator, *inputs, 0.0, 0.37, new_config, True)

    np.testing.assert_allclose(new_accumulator, old_accumulator)


def test_thermal_expectation_from_hamiltonian_matches_legacy():
    kx, ky = 0.21, -0.34
    amp = PairingAmplitudes(delta0_eV=0.04)
    delta = pairing_matrix("spm", kx, ky, amp)
    hamiltonian = bdg_hamiltonian(kx, ky, delta)
    vertex = np.eye(8, dtype=complex)
    old_config = OldKuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)
    new_config = KuboConfig(omega_eV=0.08, temperature_eV=0.02, eta_eV=1e-4)

    old = old_thermal_expectation(kx, ky, delta, vertex, old_config)
    new = thermal_expectation_bdg_from_hamiltonian(hamiltonian, vertex, new_config)

    np.testing.assert_allclose(new, old)
