"""Primitive crystal-xy Ward validation for finite-q BdG response.

This module is the production-facing Ward contract.  It never transforms the
microscopic response through LT or G/TM/TE target bases.  The longitudinal
projection is available only as an optional diagnostic view of residuals that
were already computed in the primitive crystal ``(A0, Ax, Ay)`` basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from lno327.bdg.finite_q import density_vertex
from lno327.bdg.hamiltonian import bdg_hamiltonian_from_model_pairing
from lno327.response.effective_kernel import (
    AMPLITUDE_PHASE_ORDER,
    PRIMITIVE_EM_BASIS,
    PRIMITIVE_EM_ORDER,
    EffectiveEMKernel,
)
from lno327.response.finite_q import thermal_expectation_bdg_from_hamiltonian, vertex_band
from lno327.response.finite_q_bdg import (
    bdg_contact_vertex_from_spec,
    bdg_eigensystem_from_model_pairing,
    bdg_vector_vertex_from_spec,
    require_peierls_finite_q_support,
)
from lno327.response.occupations import fermi_function
from lno327.response.validation import validate_finite_q_inputs

WARD_CONVENTION = "primitive_xy_rhs_aware_v1"
DEFAULT_RESIDUAL_TOLERANCE = 1e-9
DEFAULT_CONDITION_MAX = 1e12


def _readonly_complex_vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    vector = np.array(value, dtype=complex, copy=True).reshape(-1)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {vector.shape}")
    if not np.isfinite(vector.real).all() or not np.isfinite(vector.imag).all():
        raise ValueError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _readonly_real_vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True).reshape(-1)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _relative_residual(residual: np.ndarray, *references: np.ndarray) -> float:
    scale = max((_norm(reference) for reference in references), default=0.0)
    return _norm(residual) / max(scale, 1e-30)


@dataclass(frozen=True)
class PrimitiveWardRHS:
    """Finite-q translation/contact RHS in primitive crystal coordinates."""

    left: np.ndarray
    right: np.ndarray
    q_model: np.ndarray
    xi_eV: float
    delta0_eV: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _readonly_complex_vector(self.left, 3, "left"))
        object.__setattr__(self, "right", _readonly_complex_vector(self.right, 3, "right"))
        object.__setattr__(self, "q_model", _readonly_real_vector(self.q_model, 2, "q_model"))
        xi = float(self.xi_eV)
        delta0 = float(self.delta0_eV)
        if not np.isfinite(xi) or xi < 0.0:
            raise ValueError("xi_eV must be finite and non-negative")
        if not np.isfinite(delta0) or delta0 < 0.0:
            raise ValueError("delta0_eV must be finite and non-negative")
        object.__setattr__(self, "xi_eV", xi)
        object.__setattr__(self, "delta0_eV", delta0)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class WardSideValidation:
    """One-sided primitive and Schur-effective Ward decomposition."""

    primitive_total: np.ndarray
    primitive_rhs: np.ndarray
    primitive_residual: np.ndarray
    collective_residual: np.ndarray
    collective_projection: np.ndarray
    effective_direct: np.ndarray
    effective_predicted: np.ndarray
    effective_residual: np.ndarray
    primitive_relative_residual: float
    effective_relative_residual: float

    def __post_init__(self) -> None:
        for name in (
            "primitive_total",
            "primitive_rhs",
            "primitive_residual",
            "collective_projection",
            "effective_direct",
            "effective_predicted",
            "effective_residual",
        ):
            object.__setattr__(self, name, _readonly_complex_vector(getattr(self, name), 3, name))
        object.__setattr__(
            self,
            "collective_residual",
            _readonly_complex_vector(self.collective_residual, 2, "collective_residual"),
        )
        for name in ("primitive_relative_residual", "effective_relative_residual"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class EffectiveWardValidation:
    """RHS-aware Ward validation of one primitive effective kernel."""

    left: WardSideValidation
    right: WardSideValidation
    u_left: np.ndarray
    u_right: np.ndarray
    w_left: np.ndarray
    w_right: np.ndarray
    q_model: np.ndarray
    xi_eV: float
    delta0_eV: float
    schur_condition_number: float
    schur_inverse_method: str
    residual_tolerance: float
    condition_max: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "u_left", _readonly_complex_vector(self.u_left, 3, "u_left"))
        object.__setattr__(self, "u_right", _readonly_complex_vector(self.u_right, 3, "u_right"))
        object.__setattr__(self, "w_left", _readonly_complex_vector(self.w_left, 2, "w_left"))
        object.__setattr__(self, "w_right", _readonly_complex_vector(self.w_right, 2, "w_right"))
        object.__setattr__(self, "q_model", _readonly_real_vector(self.q_model, 2, "q_model"))
        for name in (
            "xi_eV",
            "delta0_eV",
            "schur_condition_number",
            "residual_tolerance",
            "condition_max",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "schur_inverse_method", str(self.schur_inverse_method))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def condition_ok(self) -> bool:
        return bool(
            self.schur_inverse_method == "inv"
            and self.schur_condition_number <= self.condition_max
        )

    @property
    def primitive_closed(self) -> bool:
        return bool(
            self.left.primitive_relative_residual <= self.residual_tolerance
            and self.right.primitive_relative_residual <= self.residual_tolerance
        )

    @property
    def effective_closed(self) -> bool:
        return bool(
            self.left.effective_relative_residual <= self.residual_tolerance
            and self.right.effective_relative_residual <= self.residual_tolerance
        )

    @property
    def passed(self) -> bool:
        return bool(self.condition_ok and self.primitive_closed and self.effective_closed)

    def require_passed(self) -> None:
        if not self.passed:
            raise ValueError(
                "primitive-xy RHS-aware Ward validation failed: "
                f"primitive=({self.left.primitive_relative_residual:.3e}, "
                f"{self.right.primitive_relative_residual:.3e}), "
                f"effective=({self.left.effective_relative_residual:.3e}, "
                f"{self.right.effective_relative_residual:.3e}), "
                f"condition={self.schur_condition_number:.3e}, "
                f"inverse_method={self.schur_inverse_method}"
            )


@dataclass(frozen=True)
class WardLTDiagnostics:
    """Optional LT projection of already-computed primitive-xy residuals."""

    left_primitive_residual: np.ndarray
    right_primitive_residual: np.ndarray
    left_effective_residual: np.ndarray
    right_effective_residual: np.ndarray
    xy_norms: tuple[float, float, float, float]
    lt_norms: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name in (
            "left_primitive_residual",
            "right_primitive_residual",
            "left_effective_residual",
            "right_effective_residual",
        ):
            object.__setattr__(self, name, _readonly_complex_vector(getattr(self, name), 3, name))


def primitive_ward_vectors_xy(
    xi_eV: float,
    q_model: np.ndarray,
    delta0_eV: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed primitive Ward vectors in crystal ``(A0, Ax, Ay)``."""

    xi = float(xi_eV)
    delta0 = float(delta0_eV)
    q = _readonly_real_vector(q_model, 2, "q_model")
    if xi < 0.0 or not np.isfinite(xi):
        raise ValueError("xi_eV must be finite and non-negative")
    if delta0 < 0.0 or not np.isfinite(delta0):
        raise ValueError("delta0_eV must be finite and non-negative")
    u_left = np.asarray([1j * xi, q[0], q[1]], dtype=complex)
    u_right = np.asarray([-1j * xi, q[0], q[1]], dtype=complex)
    w = np.asarray([0.0 + 0.0j, -2j * delta0], dtype=complex)
    return u_left, u_right, w.copy(), w.copy()


def primitive_ward_rhs_from_model_ansatz(
    spec: object,
    ansatz: object,
    q_model: np.ndarray,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    config: object,
    pairing_params: object,
    *,
    current_vertex: str = "peierls",
) -> PrimitiveWardRHS:
    """Evaluate the analytic translation/contact RHS on the response quadrature.

    The returned vector is ``equal_forward - delta_v_mid + qM_mid``.  Under the
    current source/observable convention the same primitive vector appears on
    the left and right identities.  Both sides are stored separately so a
    future convention change cannot silently rely on that equality.
    """

    if current_vertex != "peierls":
        raise ValueError("production Ward RHS requires current_vertex='peierls'")
    require_peierls_finite_q_support(spec)
    q, points, weights = validate_finite_q_inputs(q_model, k_points, k_weights, config)
    xi = float(config.omega_eV)
    if xi < 0.0 or not np.isfinite(xi):
        raise ValueError("config.omega_eV must be finite and non-negative")
    delta0 = float(getattr(pairing_params, "delta0_eV", 0.0))
    if delta0 < 0.0 or not np.isfinite(delta0):
        raise ValueError("pairing_params.delta0_eV must be finite and non-negative")

    qx, qy = float(q[0]), float(q[1])
    dim = np.asarray(
        spec.normal_hamiltonian(float(points[0, 0]), float(points[0, 1]))
    ).shape[0]
    rho = density_vertex(int(dim))
    equal_forward = np.zeros(3, dtype=complex)
    delta_v_mid = np.zeros(3, dtype=complex)
    q_contact_mid = np.zeros(3, dtype=complex)

    for weight, (kx_value, ky_value) in zip(weights, points, strict=True):
        kx = float(kx_value)
        ky = float(ky_value)
        delta_minus = ansatz.mean_pairing(
            kx - 0.5 * qx, ky - 0.5 * qy, pairing_params
        )
        delta_plus = ansatz.mean_pairing(
            kx + 0.5 * qx, ky + 0.5 * qy, pairing_params
        )
        delta_mid = ansatz.mean_pairing(kx, ky, pairing_params)
        bands_minus = bdg_eigensystem_from_model_pairing(
            spec, kx - 0.5 * qx, ky - 0.5 * qy, delta_minus
        )
        bands_plus = bdg_eigensystem_from_model_pairing(
            spec, kx + 0.5 * qx, ky + 0.5 * qy, delta_plus
        )
        occupations_minus = fermi_function(
            bands_minus.energies, config.fermi_level_eV, config.temperature_eV
        )
        occupations_plus = fermi_function(
            bands_plus.energies, config.fermi_level_eV, config.temperature_eV
        )

        vx = bdg_vector_vertex_from_spec(
            spec, kx, ky, qx, qy, "x", current_vertex
        )
        vy = bdg_vector_vertex_from_spec(
            spec, kx, ky, qx, qy, "y", current_vertex
        )
        source_vertices = (rho, vx, vy)
        source_band = tuple(
            vertex_band(bands_minus.states, vertex, bands_plus.states)
            for vertex in source_vertices
        )
        rho_band = source_band[0]
        for m in range(len(bands_minus.energies)):
            for n in range(len(bands_plus.energies)):
                occupation_difference = float(
                    occupations_minus[m] - occupations_plus[n]
                )
                if occupation_difference == 0.0:
                    continue
                equal_forward += (
                    0.5
                    * float(weight)
                    * np.asarray(
                        [
                            occupation_difference
                            * rho_band[m, n]
                            * np.conjugate(source_band[j][m, n])
                            for j in range(3)
                        ],
                        dtype=complex,
                    )
                )

        h_mid = bdg_hamiltonian_from_model_pairing(spec, kx, ky, delta_mid)
        for i, direction_i in enumerate(("x", "y")):
            qi = qx if direction_i == "x" else qy
            for j, direction_j in enumerate(("x", "y")):
                contact_vertex = bdg_contact_vertex_from_spec(
                    spec,
                    kx,
                    ky,
                    qx,
                    qy,
                    direction_i,
                    direction_j,
                    current_vertex,
                )
                direct = -float(weight) * thermal_expectation_bdg_from_hamiltonian(
                    h_mid, contact_vertex, config
                )
                q_contact_mid[1 + j] += qi * direct

        for j, direction in enumerate(("x", "y"), start=1):
            vertex_plus = bdg_vector_vertex_from_spec(
                spec,
                kx + 0.5 * qx,
                ky + 0.5 * qy,
                qx,
                qy,
                direction,
                current_vertex,
            )
            vertex_minus = bdg_vector_vertex_from_spec(
                spec,
                kx - 0.5 * qx,
                ky - 0.5 * qy,
                qx,
                qy,
                direction,
                current_vertex,
            )
            delta_v_mid[j] += float(weight) * thermal_expectation_bdg_from_hamiltonian(
                h_mid, vertex_plus - vertex_minus, config
            )

    rhs = equal_forward - delta_v_mid + q_contact_mid
    return PrimitiveWardRHS(
        left=rhs,
        right=rhs.copy(),
        q_model=q,
        xi_eV=xi,
        delta0_eV=delta0,
        metadata={
            "convention": WARD_CONVENTION,
            "basis": PRIMITIVE_EM_BASIS,
            "primitive_order": PRIMITIVE_EM_ORDER,
            "collective_order": AMPLITUDE_PHASE_ORDER,
            "formula": "R_S = equal_forward - delta_v_mid + qM_mid",
            "left_right_equal_under_current_convention": True,
            "equal_forward": equal_forward.copy(),
            "delta_v_mid": delta_v_mid.copy(),
            "qM_mid": q_contact_mid.copy(),
            "num_quadrature_points": int(points.shape[0]),
        },
    )


def _kernel_inverse(kernel: EffectiveEMKernel) -> np.ndarray:
    if kernel.schur_inverse_method == "inv":
        return np.linalg.inv(np.asarray(kernel.k_etaeta, dtype=complex))
    if kernel.schur_inverse_method == "pinv_diagnostic":
        return np.linalg.pinv(np.asarray(kernel.k_etaeta, dtype=complex))
    raise ValueError(
        "unsupported Schur inverse method for Ward validation: "
        f"{kernel.schur_inverse_method!r}"
    )


def validate_effective_ward_xy(
    kernel: EffectiveEMKernel,
    rhs: PrimitiveWardRHS,
    *,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
    condition_max: float = DEFAULT_CONDITION_MAX,
) -> EffectiveWardValidation:
    """Validate primitive and Schur-effective Ward identities directly in xy."""

    if not np.allclose(kernel.q_model, rhs.q_model, rtol=0.0, atol=1e-14):
        raise ValueError("kernel and Ward RHS q_model do not match")
    if not np.isclose(kernel.xi_eV, rhs.xi_eV, rtol=1e-12, atol=1e-14):
        raise ValueError("kernel and Ward RHS xi_eV do not match")
    tolerance = float(residual_tolerance)
    maximum_condition = float(condition_max)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("residual_tolerance must be finite and non-negative")
    if maximum_condition <= 0.0 or not np.isfinite(maximum_condition):
        raise ValueError("condition_max must be finite and positive")

    u_left, u_right, w_left, w_right = primitive_ward_vectors_xy(
        kernel.xi_eV, kernel.q_model, rhs.delta0_eV
    )
    inverse = _kernel_inverse(kernel)

    left_primitive_total = u_left @ kernel.k_ss + w_left @ kernel.k_etas
    left_collective = u_left @ kernel.k_seta + w_left @ kernel.k_etaeta
    left_projection = left_collective @ inverse @ kernel.k_etas
    left_effective_direct = u_left @ kernel.k_eff
    left_effective_predicted = rhs.left - left_projection
    left_primitive_residual = left_primitive_total - rhs.left
    left_effective_residual = left_effective_direct - left_effective_predicted

    right_primitive_total = kernel.k_ss @ u_right + kernel.k_seta @ w_right
    right_collective = kernel.k_etas @ u_right + kernel.k_etaeta @ w_right
    right_projection = kernel.k_seta @ inverse @ right_collective
    right_effective_direct = kernel.k_eff @ u_right
    right_effective_predicted = rhs.right - right_projection
    right_primitive_residual = right_primitive_total - rhs.right
    right_effective_residual = right_effective_direct - right_effective_predicted

    left = WardSideValidation(
        primitive_total=left_primitive_total,
        primitive_rhs=rhs.left,
        primitive_residual=left_primitive_residual,
        collective_residual=left_collective,
        collective_projection=left_projection,
        effective_direct=left_effective_direct,
        effective_predicted=left_effective_predicted,
        effective_residual=left_effective_residual,
        primitive_relative_residual=_relative_residual(
            left_primitive_residual, left_primitive_total, rhs.left
        ),
        effective_relative_residual=_relative_residual(
            left_effective_residual,
            left_effective_direct,
            left_effective_predicted,
            rhs.left,
        ),
    )
    right = WardSideValidation(
        primitive_total=right_primitive_total,
        primitive_rhs=rhs.right,
        primitive_residual=right_primitive_residual,
        collective_residual=right_collective,
        collective_projection=right_projection,
        effective_direct=right_effective_direct,
        effective_predicted=right_effective_predicted,
        effective_residual=right_effective_residual,
        primitive_relative_residual=_relative_residual(
            right_primitive_residual, right_primitive_total, rhs.right
        ),
        effective_relative_residual=_relative_residual(
            right_effective_residual,
            right_effective_direct,
            right_effective_predicted,
            rhs.right,
        ),
    )
    condition = (
        float(kernel.schur_condition_number)
        if kernel.schur_condition_number is not None
        else float(np.linalg.cond(kernel.k_etaeta))
    )
    return EffectiveWardValidation(
        left=left,
        right=right,
        u_left=u_left,
        u_right=u_right,
        w_left=w_left,
        w_right=w_right,
        q_model=kernel.q_model,
        xi_eV=kernel.xi_eV,
        delta0_eV=rhs.delta0_eV,
        schur_condition_number=condition,
        schur_inverse_method=kernel.schur_inverse_method,
        residual_tolerance=tolerance,
        condition_max=maximum_condition,
        metadata={
            "convention": WARD_CONVENTION,
            "basis": PRIMITIVE_EM_BASIS,
            "primitive_order": PRIMITIVE_EM_ORDER,
            "collective_order": AMPLITUDE_PHASE_ORDER,
            "zero_rhs_check_is_invalid_at_finite_q": True,
            "lt_projection_is_diagnostic_only": True,
        },
    )


def project_ward_validation_xy_to_lt(
    validation: EffectiveWardValidation,
) -> WardLTDiagnostics:
    """Project xy residuals to LT without recomputing any Ward identity."""

    q = np.asarray(validation.q_model, dtype=float)
    magnitude = float(np.linalg.norm(q))
    if magnitude == 0.0:
        raise ValueError("q_model must be nonzero to define the optional LT diagnostic")
    qhat = q / magnitude
    projection = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, qhat[0], qhat[1]],
            [0.0, -qhat[1], qhat[0]],
        ],
        dtype=float,
    )
    left_primitive = validation.left.primitive_residual @ projection.T
    right_primitive = projection @ validation.right.primitive_residual
    left_effective = validation.left.effective_residual @ projection.T
    right_effective = projection @ validation.right.effective_residual
    xy_norms = (
        _norm(validation.left.primitive_residual),
        _norm(validation.right.primitive_residual),
        _norm(validation.left.effective_residual),
        _norm(validation.right.effective_residual),
    )
    lt_norms = (
        _norm(left_primitive),
        _norm(right_primitive),
        _norm(left_effective),
        _norm(right_effective),
    )
    return WardLTDiagnostics(
        left_primitive_residual=left_primitive,
        right_primitive_residual=right_primitive,
        left_effective_residual=left_effective,
        right_effective_residual=right_effective,
        xy_norms=xy_norms,
        lt_norms=lt_norms,
    )


__all__ = [
    "DEFAULT_CONDITION_MAX",
    "DEFAULT_RESIDUAL_TOLERANCE",
    "EffectiveWardValidation",
    "PrimitiveWardRHS",
    "WARD_CONVENTION",
    "WardLTDiagnostics",
    "WardSideValidation",
    "primitive_ward_rhs_from_model_ansatz",
    "primitive_ward_vectors_xy",
    "project_ward_validation_xy_to_lt",
    "validate_effective_ward_xy",
]
