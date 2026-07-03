"""Collective pairing vertices for the LNO327 four-orbital model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lno327.models.lno327_four_orbital.bdg import bdg_hamiltonian
from lno327.models.lno327_four_orbital.pairing import bond_endpoint_gauge_form_factor, pairing_matrix
from lno327.models.lno327_four_orbital.parameters import PairingAmplitudes, PairingAnsatzName, PhaseVertexName


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

        from lno327.conductivity import fermi_function

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
