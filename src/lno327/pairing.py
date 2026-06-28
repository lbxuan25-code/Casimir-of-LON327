"""s_pm and d-wave superconducting pairing ansaetze in eV."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .model import normal_state_hamiltonian

NormalStateBuilder = Callable[[float, float], np.ndarray]
PairingKind = Literal["spm", "dwave"]
PairingAnsatzName = Literal["onsite_s", "spm", "dwave"]
PhaseVertexName = Literal["midpoint", "symmetric_kpm", "bond_endpoint_gauge"]
PairingBondStatus = Literal["PASSED", "FAILED", "PAIRING_BOND_REPRESENTATION_UNAVAILABLE"]


@dataclass(frozen=True)
class PairingAmplitudes:
    """Pairing seed amplitude in eV.

    The default is a small algebraic seed, not a fitted superconducting gap.
    """

    delta0_eV: float = 0.04

    @property
    def delta0(self) -> float:
        """Backward-compatible alias for the eV pairing amplitude."""

        return self.delta0_eV


def spm_pairing_matrix(kx: float, ky: float, amp: PairingAmplitudes | None = None) -> np.ndarray:
    """Return the 4x4 bilayer dz2 interlayer s_pm pairing matrix in eV.

    The basis is ``(dz1, dx1, dz2, dx2)``. Only the two dz2-like orbitals
    pair across layers, representing sign-changing bonding/antibonding
    bilayer s_pm pairing.
    """

    amp = amp or PairingAmplitudes()
    return np.array(
        [
            [0.0, 0.0, amp.delta0_eV, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [amp.delta0_eV, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=complex,
    )


def dwave_pairing_matrix(kx: float, ky: float, amp: PairingAmplitudes | None = None) -> np.ndarray:
    """Return the 4x4 same-layer dz2-dx2_y2 B1g pairing matrix in eV.

    The explicit momentum factor is A1g, ``cos(kx) + cos(ky)``. Combined with
    the B1g symmetry of the dx2_y2 orbital, the total pairing transforms as
    the d-wave/B1g channel.
    """

    amp = amp or PairingAmplitudes()
    form = np.cos(kx) + np.cos(ky)
    orbital_structure = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=complex,
    )
    return amp.delta0_eV * form * orbital_structure


def pairing_matrix(
    kind: PairingKind,
    kx: float,
    ky: float,
    amp: PairingAmplitudes | None = None,
) -> np.ndarray:
    """Return the requested 4x4 pairing matrix in eV."""

    if kind == "spm":
        return spm_pairing_matrix(kx, ky, amp)
    if kind == "dwave":
        return dwave_pairing_matrix(kx, ky, amp)
    raise ValueError("pairing kind must be 'spm' or 'dwave'")


def bdg_hamiltonian(
    kx: float,
    ky: float,
    pairing: np.ndarray,
    normal_state: NormalStateBuilder = normal_state_hamiltonian,
) -> np.ndarray:
    """Build an 8x8 BdG Hamiltonian from a 4x4 normal state and pairing matrix."""

    h_k = normal_state(kx, ky)
    h_minus_k = normal_state(-kx, -ky)
    pairing = np.asarray(pairing)

    if h_k.shape != (4, 4) or h_minus_k.shape != (4, 4):
        raise ValueError("normal_state must return a 4x4 matrix")
    if pairing.shape != (4, 4):
        raise ValueError("pairing must be a 4x4 matrix")
    if not np.allclose(h_k, h_k.conjugate().T):
        raise ValueError("normal_state(k) must be Hermitian")
    if not np.allclose(h_minus_k, h_minus_k.conjugate().T):
        raise ValueError("normal_state(-k) must be Hermitian")

    return np.block(
        [
            [h_k, pairing],
            [pairing.conjugate().T, -h_minus_k.T],
        ]
    )


@dataclass(frozen=True)
class PairingBond:
    channel: str
    left_orbital: int
    right_orbital: int
    left_offset: tuple[float, float]
    right_offset: tuple[float, float]
    coefficient: complex
    matrix_block: str | None = None

    @property
    def displacement(self) -> np.ndarray:
        return np.asarray(self.right_offset, dtype=float) - np.asarray(self.left_offset, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (np.asarray(self.right_offset, dtype=float) + np.asarray(self.left_offset, dtype=float))


def pairing_bond_list(pairing: str, amp: PairingAmplitudes | None = None) -> list[PairingBond]:
    """Return the exact finite-Fourier pairing bonds used by the current code."""

    amplitude = amp or PairingAmplitudes()
    delta0 = complex(amplitude.delta0_eV)
    if pairing == "onsite_s":
        return [
            PairingBond("onsite_s", orbital, orbital, (0.0, 0.0), (0.0, 0.0), delta0, "onsite_identity")
            for orbital in range(4)
        ]
    if pairing == "spm":
        return [
            PairingBond("spm", 0, 2, (0.0, 0.0), (0.0, 0.0), delta0, "interlayer_dz2"),
            PairingBond("spm", 2, 0, (0.0, 0.0), (0.0, 0.0), delta0, "interlayer_dz2"),
        ]
    if pairing == "dwave":
        bonds: list[PairingBond] = []
        for left, right in ((0, 1), (1, 0), (2, 3), (3, 2)):
            for offset in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                half = (0.5 * offset[0], 0.5 * offset[1])
                bonds.append(
                    PairingBond(
                        "dwave",
                        left,
                        right,
                        (-half[0], -half[1]),
                        half,
                        0.5 * delta0,
                        "same_layer_interorbital_A1g_cos_sum",
                    )
                )
        return bonds
    return []


def pairing_bond_status(pairing: str) -> PairingBondStatus:
    return "PAIRING_BOND_REPRESENTATION_UNAVAILABLE" if not pairing_bond_list(pairing) else "PASSED"


def pairing_from_bonds(
    pairing: str,
    kx: float,
    ky: float,
    amp: PairingAmplitudes | None = None,
) -> np.ndarray:
    """Reconstruct Delta(k) = sum_b c_b exp(-i k.d_b) P_b from pairing bonds."""

    output = np.zeros((4, 4), dtype=complex)
    k = np.asarray([kx, ky], dtype=float)
    for bond in pairing_bond_list(pairing, amp):
        output[bond.left_orbital, bond.right_orbital] += bond.coefficient * np.exp(
            -1j * float(np.dot(k, bond.displacement))
        )
    return output


def bond_center_form_factor(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    amp: PairingAmplitudes,
) -> np.ndarray:
    """Return the bond-center amplitude form factor Delta_bond(k,q)/delta0."""

    delta0 = float(amp.delta0_eV)
    if delta0 == 0.0:
        raise ValueError("pairing bond form factor is undefined for delta0=0")
    output = np.zeros((4, 4), dtype=complex)
    k = np.asarray([kx, ky], dtype=float)
    q = np.asarray([qx, qy], dtype=float)
    for bond in pairing_bond_list(pairing, amp):
        output[bond.left_orbital, bond.right_orbital] += (
            bond.coefficient
            * np.exp(-1j * float(np.dot(k, bond.displacement)))
            * np.exp(1j * float(np.dot(q, bond.center)))
            / delta0
        )
    return output


def bond_endpoint_gauge_form_factor(
    pairing: str,
    kx: float,
    ky: float,
    qx: float,
    qy: float,
    amp: PairingAmplitudes,
) -> np.ndarray:
    """Return the endpoint-average gauge-phase form factor from pairing bonds."""

    delta0 = float(amp.delta0_eV)
    if delta0 == 0.0:
        raise ValueError("pairing bond form factor is undefined for delta0=0")
    output = np.zeros((4, 4), dtype=complex)
    k = np.asarray([kx, ky], dtype=float)
    q = np.asarray([qx, qy], dtype=float)
    for bond in pairing_bond_list(pairing, amp):
        displacement = bond.displacement
        output[bond.left_orbital, bond.right_orbital] += (
            bond.coefficient
            * np.exp(-1j * float(np.dot(k, displacement)))
            * np.exp(1j * float(np.dot(q, bond.center)))
            * np.cos(0.5 * float(np.dot(q, displacement)))
            / delta0
        )
    return output


def _amplitude_vertex(phi: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi)
    return np.block([[zero, phi], [phi.conjugate().T, zero]]).astype(complex)


def _eta2_phase_vertex(phi: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi)
    return np.block([[zero, 1j * phi], [-1j * phi.conjugate().T, zero]]).astype(complex)


def _kubo_static_factor(
    em: float,
    en: float,
    fm: float,
    fn: float,
    config,
) -> complex:
    delta_e = float(em) - float(en)
    if abs(delta_e) < config.eta_eV:
        shifted = float(em) - float(config.fermi_level_eV)
        if config.temperature_eV <= 0.0:
            width = max(float(config.eta_eV), 1e-12)
            return -float(width / (np.pi * (shifted**2 + width**2)))
        x = np.clip(shifted / (2.0 * config.temperature_eV), -350.0, 350.0)
        return -float(1.0 / (4.0 * config.temperature_eV * np.cosh(x) ** 2))
    return (float(fm) - float(fn)) / delta_e


@dataclass(frozen=True)
class PairingAnsatz:
    """Minimal separable fixed-form-factor ansatz."""

    name: PairingAnsatzName
    phase_vertex: PhaseVertexName = "bond_endpoint_gauge"
    channel_names: tuple[str, ...] = ("eta1", "eta2")

    def mean_pairing(self, kx: float, ky: float, amp: PairingAmplitudes | None = None) -> np.ndarray:
        amplitude = amp or PairingAmplitudes()
        if self.name == "onsite_s":
            return amplitude.delta0_eV * np.eye(4, dtype=complex)
        return pairing_matrix(self.name, kx, ky, amplitude)

    def collective_form_factor(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        amp: PairingAmplitudes,
    ) -> np.ndarray:
        delta0 = float(amp.delta0_eV)
        if delta0 == 0.0:
            raise ValueError("pairing form factor is undefined for delta0=0")
        if self.phase_vertex == "midpoint":
            return self.mean_pairing(kx, ky, amp) / delta0
        if self.phase_vertex == "symmetric_kpm":
            phi_minus = self.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp) / delta0
            phi_plus = self.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp) / delta0
            return 0.5 * (phi_minus + phi_plus)
        if self.phase_vertex == "bond_endpoint_gauge":
            return bond_endpoint_gauge_form_factor(self.name, kx, ky, qx, qy, amp)
        raise ValueError("phase_vertex must be 'midpoint', 'symmetric_kpm', or 'bond_endpoint_gauge'")

    def phase_pairing_matrix(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        amp: PairingAmplitudes,
    ) -> np.ndarray:
        if self.phase_vertex == "midpoint":
            return self.mean_pairing(kx, ky, amp)
        if self.phase_vertex == "symmetric_kpm":
            delta_minus = self.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp)
            delta_plus = self.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp)
            return 0.5 * (delta_minus + delta_plus)
        if float(amp.delta0_eV) == 0.0:
            return np.zeros((4, 4), dtype=complex)
        return float(amp.delta0_eV) * self.collective_form_factor(kx, ky, qx, qy, amp)

    def collective_vertices(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        amp: PairingAmplitudes,
    ) -> tuple[np.ndarray, ...]:
        phi = self.collective_form_factor(kx, ky, qx, qy, amp)
        return (_amplitude_vertex(phi), _eta2_phase_vertex(phi))

    def hs_counterterm(
        self,
        config,
        k_points: np.ndarray,
        k_weights: np.ndarray,
        amp: PairingAmplitudes,
    ) -> np.ndarray:
        """Return the current q=0 Goldstone gap-equation counterterm matrix."""

        from .conductivity import fermi_function

        points = np.asarray(k_points, dtype=float)
        weights = np.asarray(k_weights, dtype=float)
        bubble = np.zeros((2, 2), dtype=complex)
        for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
            kx = float(kx_value)
            ky = float(ky_value)
            delta = self.mean_pairing(kx, ky, amp)
            energies, states = np.linalg.eigh(bdg_hamiltonian(kx, ky, delta))
            occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
            vertices = self.collective_vertices(kx, ky, 0.0, 0.0, amp)
            band = tuple(states.conjugate().T @ vertex @ states for vertex in vertices)
            for m, energy_m in enumerate(energies):
                for n, energy_n in enumerate(energies):
                    factor = 0.5 * float(weight) * _kubo_static_factor(
                        float(energy_m),
                        float(energy_n),
                        float(occupations[m]),
                        float(occupations[n]),
                        config,
                    )
                    for left_idx, left in enumerate(band):
                        for right_idx, right in enumerate(band):
                            bubble[left_idx, right_idx] += factor * left[m, n] * np.conjugate(right[m, n])
        counterterm = np.zeros((len(self.channel_names), len(self.channel_names)), dtype=complex)
        counterterm[:, :] = -complex(bubble[1, 1]) * np.eye(len(self.channel_names), dtype=complex)
        return counterterm

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channel_names": list(self.channel_names),
            "phase_vertex": self.phase_vertex,
            "model": "minimal_separable_fixed_form_factor",
            "bond_resolved_extra_modes": False,
            "xi_omitted_modes": False,
        }


def build_pairing_ansatz(name: PairingAnsatzName, *, phase_vertex: PhaseVertexName = "bond_endpoint_gauge") -> PairingAnsatz:
    if name not in {"onsite_s", "spm", "dwave"}:
        raise ValueError("pairing ansatz must be 'onsite_s', 'spm', or 'dwave'")
    return PairingAnsatz(name=name, phase_vertex=phase_vertex)
