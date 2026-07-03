"""Static Matsubara response policies before formal Casimir calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bdg_response import bdg_total_kernel_imag_axis
from .conductivity import KuboConfig, bosonic_matsubara_energy_eV, k_weights as normalized_k_weights, uniform_bz_mesh
from .models.lno327_four_orbital.parameters import PairingAmplitudes
from .response_interface import LocalSheetResponse, ResponseKind, local_response_imag_axis

StaticResponsePolicy = Literal["skip", "extrapolate_from_lowest_matsubara", "use_static_kernel"]


@dataclass(frozen=True)
class StaticResponseResult:
    """Result of applying an n=0 Matsubara policy."""

    kind: ResponseKind
    n: int
    omega_eV: float
    policy: StaticResponsePolicy
    status: str
    approximate: bool
    matrix: np.ndarray | None
    response: LocalSheetResponse | None
    unit_label: str
    source: str
    valid_for_casimir_input: bool
    notes: tuple[str, ...]


def _mesh_and_weights(
    nk: int | None,
    k_points: Sequence[tuple[float, float]] | np.ndarray | None,
    weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if k_points is None:
        if nk is None:
            raise ValueError("Either nk or k_points must be provided")
        points = uniform_bz_mesh(nk)
    else:
        points = np.asarray(k_points, dtype=float)
    if weights is None:
        return points, normalized_k_weights(points)
    return points, np.asarray(weights, dtype=float)


def local_response_matsubara_index(
    kind: ResponseKind,
    n: int,
    temperature_K: float,
    policy: StaticResponsePolicy = "skip",
    nk: int | None = None,
    k_points: Sequence[tuple[float, float]] | np.ndarray | None = None,
    k_weights: Sequence[float] | np.ndarray | None = None,
    eta_eV: float = 1e-4,
    pairing_params: PairingAmplitudes | None = None,
) -> StaticResponseResult:
    """Return response information for a Matsubara index with explicit n=0 policy."""

    if n < 0:
        raise ValueError("n must be non-negative")
    if policy not in {"skip", "extrapolate_from_lowest_matsubara", "use_static_kernel"}:
        raise ValueError("Unknown static response policy")
    points, weights = _mesh_and_weights(nk, k_points, k_weights)

    if n >= 1:
        omega = bosonic_matsubara_energy_eV(n, temperature_K)
        response = local_response_imag_axis(
            kind,
            omega,
            points,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            pairing_params=pairing_params,
            k_weights=weights,
        )
        return StaticResponseResult(
            kind=kind,
            n=n,
            omega_eV=omega,
            policy=policy,
            status="finite_matsubara",
            approximate=False,
            matrix=response.matrix,
            response=response,
            unit_label=response.unit_label,
            source=response.source,
            valid_for_casimir_input=False,
            notes=response.notes + ("n >= 1 uses existing local response definition",),
        )

    if policy == "skip":
        return StaticResponseResult(
            kind=kind,
            n=0,
            omega_eV=0.0,
            policy=policy,
            status="skipped",
            approximate=False,
            matrix=None,
            response=None,
            unit_label="unresolved_n0",
            source="static_response_policy",
            valid_for_casimir_input=False,
            notes=("n=0 skipped by policy", "Sigma_SC direct n=0 division is not allowed"),
        )

    if policy == "extrapolate_from_lowest_matsubara":
        omega = bosonic_matsubara_energy_eV(1, temperature_K)
        response = local_response_imag_axis(
            kind,
            omega,
            points,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            pairing_params=pairing_params,
            k_weights=weights,
        )
        return StaticResponseResult(
            kind=kind,
            n=0,
            omega_eV=0.0,
            policy=policy,
            status="extrapolated",
            approximate=True,
            matrix=response.matrix,
            response=response,
            unit_label=response.unit_label,
            source=response.source,
            valid_for_casimir_input=False,
            notes=response.notes
            + (
                "n=0 approximated by the lowest nonzero Matsubara response",
                "approximate=True",
            ),
        )

    if kind == "normal":
        response = local_response_imag_axis(
            kind,
            0.0,
            points,
            temperature_K=temperature_K,
            eta_eV=eta_eV,
            pairing_params=pairing_params,
            k_weights=weights,
        )
        return StaticResponseResult(
            kind=kind,
            n=0,
            omega_eV=0.0,
            policy=policy,
            status="static_kernel",
            approximate=False,
            matrix=response.matrix,
            response=response,
            unit_label=response.unit_label,
            source=response.source,
            valid_for_casimir_input=False,
            notes=response.notes + ("normal-state n=0 retained as diagnostic only",),
        )

    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=temperature_K,
        eta_eV=eta_eV,
        output_si=False,
    )
    components = bdg_total_kernel_imag_axis(
        points,
        config,
        kind,
        pairing_params,
        weights,
    )
    return StaticResponseResult(
        kind=kind,
        n=0,
        omega_eV=0.0,
        policy=policy,
        status="static_kernel",
        approximate=False,
        matrix=components.total,
        response=None,
        unit_label="model_units_BdG_Ktotal_static_diagnostic",
        source="bdg_total_kernel_imag_axis_at_zero_frequency",
        valid_for_casimir_input=False,
        notes=(
            "BdG n=0 uses K_total(0) diagnostic, not Sigma_SC=K_total/omega",
            "static kernel is not a final Casimir input",
        ),
    )
