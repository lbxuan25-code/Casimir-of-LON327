"""Collective pairing adapter for two-band validation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from lno327.models.symmetry_bdg_2band.bdg import assemble_bdg_hamiltonian
from lno327.models.symmetry_bdg_2band.normal import TAU0, TAUX, normal_hamiltonian
from lno327.models.symmetry_bdg_2band.pairing import d_wave_form_factor
from lno327.models.symmetry_bdg_2band.parameters import PairingChannel, TwoBandParameters
from lno327.response.occupations import fermi_function

PhaseVertexName = Literal["midpoint", "symmetric_kpm", "bond_endpoint_gauge"]


@dataclass(frozen=True)
class SymmetryTwoBandPairingAmplitudes:
    delta0_eV: float = 0.1
    delta_s: float | None = None
    delta_d: float | None = None

    @property
    def spm_delta(self) -> float:
        return float(self.delta0_eV if self.delta_s is None else self.delta_s)

    @property
    def dwave_delta(self) -> float:
        return float(self.delta0_eV if self.delta_d is None else self.delta_d)


def _amplitude_vertex(phi: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi)
    return np.block([[zero, phi], [phi.conjugate().T, zero]]).astype(complex)


def _eta2_phase_vertex(phi: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi)
    return np.block([[zero, 1j * phi], [-1j * phi.conjugate().T, zero]]).astype(complex)


def _amplitude_vertex_from_endpoints(phi_minus: np.ndarray, phi_plus: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi_plus)
    return np.block([[zero, phi_plus], [phi_minus.conjugate().T, zero]]).astype(complex)


def _eta2_phase_vertex_from_endpoints(phi_minus: np.ndarray, phi_plus: np.ndarray) -> np.ndarray:
    zero = np.zeros_like(phi_plus)
    return np.block([[zero, 1j * phi_plus], [-1j * phi_minus.conjugate().T, zero]]).astype(complex)


def _kubo_static_factor(em: float, en: float, fm: float, fn: float, config) -> complex:
    delta_e = float(em) - float(en)
    if abs(delta_e) < config.eta_eV:
        shifted = float(em) - float(config.fermi_level_eV)
        if config.temperature_eV <= 0.0:
            width = max(float(config.eta_eV), 1e-12)
            return -float(width / (np.pi * (shifted**2 + width**2)))
        x = np.clip(shifted / (2.0 * config.temperature_eV), -350.0, 350.0)
        return -float(1.0 / (4.0 * config.temperature_eV * np.cosh(x) ** 2))
    return (float(fm) - float(fn)) / delta_e


def _form_factor(name: PairingChannel, kx: float, ky: float) -> np.ndarray:
    if name == "normal":
        return np.zeros((2, 2), dtype=complex)
    if name == "spm":
        return TAUX.astype(complex)
    if name == "dwave":
        return d_wave_form_factor(kx, ky) * TAU0.astype(complex)
    raise ValueError("pairing ansatz must be 'normal', 'spm', or 'dwave'")


@dataclass(frozen=True)
class SymmetryTwoBandPairingAnsatz:
    name: PairingChannel
    phase_vertex: PhaseVertexName = "bond_endpoint_gauge"
    channel_names: tuple[str, ...] = ("eta1", "eta2")

    def mean_pairing(
        self,
        kx: float,
        ky: float,
        amp: SymmetryTwoBandPairingAmplitudes | None = None,
    ) -> np.ndarray:
        amplitude = amp or SymmetryTwoBandPairingAmplitudes()
        if self.name == "normal":
            return np.zeros((2, 2), dtype=complex)
        if self.name == "spm":
            return amplitude.spm_delta * TAUX.astype(complex)
        if self.name == "dwave":
            return amplitude.dwave_delta * d_wave_form_factor(kx, ky) * TAU0.astype(complex)
        raise ValueError("pairing ansatz must be 'normal', 'spm', or 'dwave'")

    def collective_form_factor(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        amp: SymmetryTwoBandPairingAmplitudes,
    ) -> np.ndarray:
        delta0 = float(amp.delta0_eV)
        if delta0 == 0.0:
            raise ValueError("pairing form factor is undefined for delta0=0")
        if self.name == "normal":
            return np.zeros((2, 2), dtype=complex)
        if self.phase_vertex == "midpoint":
            return self.mean_pairing(kx, ky, amp) / delta0
        if self.phase_vertex in {"symmetric_kpm", "bond_endpoint_gauge"}:
            phi_minus = self.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp) / delta0
            phi_plus = self.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp) / delta0
            return 0.5 * (phi_minus + phi_plus)
        raise ValueError("phase_vertex must be 'midpoint', 'symmetric_kpm', or 'bond_endpoint_gauge'")

    def phase_pairing_matrix(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        amp: SymmetryTwoBandPairingAmplitudes,
    ) -> np.ndarray:
        if self.name == "normal" or float(amp.delta0_eV) == 0.0:
            return np.zeros((2, 2), dtype=complex)
        return float(amp.delta0_eV) * self.collective_form_factor(kx, ky, qx, qy, amp)

    def _endpoint_form_factors(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        amp: SymmetryTwoBandPairingAmplitudes,
    ) -> tuple[np.ndarray, np.ndarray]:
        delta0 = float(amp.delta0_eV)
        if delta0 == 0.0:
            raise ValueError("pairing form factor is undefined for delta0=0")
        phi_minus = self.mean_pairing(kx - 0.5 * qx, ky - 0.5 * qy, amp) / delta0
        phi_plus = self.mean_pairing(kx + 0.5 * qx, ky + 0.5 * qy, amp) / delta0
        return phi_minus, phi_plus

    def collective_vertices(
        self,
        kx: float,
        ky: float,
        qx: float,
        qy: float,
        amp: SymmetryTwoBandPairingAmplitudes,
    ) -> tuple[np.ndarray, ...]:
        """Return amplitude/eta2 vertices using the Ward gauge source convention.

        Finite-q charge Ward closure has anomalous source Delta(k-q/2)+Delta(k+q/2).
        Therefore bond_endpoint_gauge uses the same endpoint-average form factor as
        phase_pairing_matrix.  Endpoint-asymmetric helpers are retained only for
        diagnostics and are not used as the eta2 Ward gauge channel.
        """
        phi = self.collective_form_factor(kx, ky, qx, qy, amp)
        return (_amplitude_vertex(phi), _eta2_phase_vertex(phi))

    def hs_counterterm(
        self,
        config,
        k_points: np.ndarray,
        k_weights: np.ndarray,
        amp: SymmetryTwoBandPairingAmplitudes,
    ) -> np.ndarray:
        points = np.asarray(k_points, dtype=float)
        weights = np.asarray(k_weights, dtype=float)
        bubble = np.zeros((2, 2), dtype=complex)
        params = TwoBandParameters(delta_s=amp.spm_delta, delta_d=amp.dwave_delta)
        for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
            kx = float(kx_value)
            ky = float(ky_value)
            delta = self.mean_pairing(kx, ky, amp)
            h = assemble_bdg_hamiltonian(
                normal_hamiltonian(kx, ky, params),
                normal_hamiltonian(-kx, -ky, params),
                delta,
            )
            energies, states = np.linalg.eigh(h)
            occupations = fermi_function(energies, config.fermi_level_eV, config.temperature_eV)
            band = tuple(states.conjugate().T @ vertex @ states for vertex in self.collective_vertices(kx, ky, 0.0, 0.0, amp))
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
            "eta2_gauge_source": "endpoint_average_delta_minus_plus_delta_plus_over_2delta0",
            "bond_endpoint_gauge_eta2_convention": "endpoint_average_not_endpoint_asymmetric",
            "model": "symmetry_bdg_2band",
            "bond_resolved_extra_modes": False,
            "validation_only": True,
            "valid_for_casimir_input": False,
        }


def build_pairing_ansatz(
    name: PairingChannel,
    *,
    phase_vertex: PhaseVertexName = "bond_endpoint_gauge",
) -> SymmetryTwoBandPairingAnsatz:
    if name not in {"normal", "spm", "dwave"}:
        raise ValueError("pairing ansatz must be 'normal', 'spm', or 'dwave'")
    return SymmetryTwoBandPairingAnsatz(name=name, phase_vertex=phase_vertex)
