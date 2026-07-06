"""Diagnostic-only finite-q Ward closure triage helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.collective.validation import validate_physical_ward_identity
from lno327.response.normal_density_current import (
    normal_physical_density_current_response_components_imag_axis_from_model,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


COMPONENT_LABELS = ("density", "current_x", "current_y")
_TOLERANCE = 1e-8
_EPS = 1e-30


@dataclass(frozen=True)
class _ResidualSummary:
    vector: np.ndarray
    norm: float
    max_norm: float
    dominant_component: str


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "valid_for_casimir_input": False,
    }


def _dominant_component(vector: np.ndarray) -> str:
    flat = np.asarray(vector, dtype=complex).reshape(-1)
    if flat.size == 0:
        return "unknown"
    index = int(np.argmax(np.abs(flat)))
    if flat.size == len(COMPONENT_LABELS):
        return COMPONENT_LABELS[index]
    if flat.size == 2 * len(COMPONENT_LABELS):
        side = "left" if index < len(COMPONENT_LABELS) else "right"
        label = COMPONENT_LABELS[index % len(COMPONENT_LABELS)]
        return f"{side}_{label}"
    return f"flat_index_{index}"


def _ward_residual_summary(matrix: np.ndarray, omega_eV: float, q_model: tuple[float, float]) -> _ResidualSummary:
    ward = validate_physical_ward_identity(
        np.asarray(matrix, dtype=complex),
        float(omega_eV),
        np.asarray(q_model, dtype=float),
        tolerance=_TOLERANCE,
    )
    vector = np.concatenate(
        [
            np.asarray(ward.left_residual, dtype=complex).reshape(-1),
            np.asarray(ward.right_residual, dtype=complex).reshape(-1),
        ]
    )
    return _ResidualSummary(
        vector=vector,
        norm=float(np.linalg.norm(vector)),
        max_norm=float(max(ward.left_norm, ward.right_norm)),
        dominant_component=_dominant_component(vector),
    )


def _summary_payload(summary: _ResidualSummary) -> dict[str, Any]:
    return {
        "max_norm": summary.max_norm,
        "vector_norm": summary.norm,
        "dominant_component": summary.dominant_component,
    }


def _complex_component_payload(vector: np.ndarray) -> list[dict[str, float | str]]:
    array = np.asarray(vector, dtype=complex).reshape(-1)
    return [
        {
            "component": COMPONENT_LABELS[index] if index < len(COMPONENT_LABELS) else f"component_{index}",
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
        }
        for index, value in enumerate(array)
    ]


def _ward_residual_detail(matrix: np.ndarray, omega_eV: float, q_model: tuple[float, float]) -> dict[str, Any]:
    ward = validate_physical_ward_identity(
        np.asarray(matrix, dtype=complex),
        float(omega_eV),
        np.asarray(q_model, dtype=float),
        tolerance=_TOLERANCE,
    )
    left = np.asarray(ward.left_residual, dtype=complex).reshape(-1)
    right = np.asarray(ward.right_residual, dtype=complex).reshape(-1)
    return {
        "left_residual_vector": _complex_component_payload(left),
        "right_residual_vector": _complex_component_payload(right),
        "left_norm": float(ward.left_norm),
        "right_norm": float(ward.right_norm),
        "dominant_left_component": _dominant_component(left),
        "dominant_right_component": _dominant_component(right),
        "max_norm": float(max(ward.left_norm, ward.right_norm)),
        "vector_norm": float(np.linalg.norm(np.concatenate([left, right]))),
        "dominant_component": _dominant_component(np.concatenate([left, right])),
    }


def _find_component_matrix(components: dict[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in components:
            return np.asarray(components[name], dtype=complex)
    return None


def _call_first(candidates: tuple[Callable[[], Any], ...]) -> Any:
    errors: list[str] = []
    for candidate in candidates:
        try:
            return candidate()
        except (AttributeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    raise AttributeError("; ".join(error for error in errors if error) or "no compatible API candidate")


def _normal_hamiltonian(spec: Any, k: np.ndarray) -> np.ndarray:
    kx = float(k[0])
    ky = float(k[1])
    return np.asarray(
        _call_first(
            (
                lambda: spec.hamiltonian(kx, ky),
                lambda: spec.hamiltonian(k),
                lambda: spec.normal_hamiltonian(kx, ky),
                lambda: spec.normal_hamiltonian(k),
                lambda: spec.h0(kx, ky),
                lambda: spec.h0(k),
            )
        ),
        dtype=complex,
    )


def _normal_vector_vertex(spec: Any, k: np.ndarray, q: np.ndarray, direction: int) -> np.ndarray:
    kx = float(k[0])
    ky = float(k[1])
    qx = float(q[0])
    qy = float(q[1])
    direction_label = ("x", "y")[direction]
    return np.asarray(
        _call_first(
            (
                lambda: spec.peierls_hamiltonian_vector_vertex(kx, ky, qx, qy, direction_label),
                lambda: spec.peierls_hamiltonian_vector_vertex(
                    kx,
                    ky,
                    qx,
                    qy,
                    direction_label,
                    hopping_terms=spec.hopping_terms(),
                ),
                lambda: spec.velocity_operator(kx, ky, direction_label),
                lambda: spec.current_vertex(direction, kx, ky, qx, qy),
                lambda: spec.current_vertex(direction_label, kx, ky, qx, qy),
                lambda: spec.current_vertex(kx, ky, qx, qy, direction),
                lambda: spec.current_vertex(kx, ky, qx, qy, direction_label),
                lambda: spec.current_vertex(k, q, direction),
                lambda: spec.current_vertex(k, q, direction_label),
                lambda: spec.vector_vertex(direction, kx, ky, qx, qy),
                lambda: spec.vector_vertex(direction_label, kx, ky, qx, qy),
                lambda: spec.vector_vertex(kx, ky, qx, qy, direction),
                lambda: spec.vector_vertex(kx, ky, qx, qy, direction_label),
                lambda: spec.vector_vertex(k, q, direction),
                lambda: spec.vector_vertex(k, q, direction_label),
                lambda: spec.peierls_vector_vertex(direction, kx, ky, qx, qy),
                lambda: spec.peierls_vector_vertex(direction_label, kx, ky, qx, qy),
                lambda: spec.peierls_vector_vertex(kx, ky, qx, qy, direction),
                lambda: spec.peierls_vector_vertex(kx, ky, qx, qy, direction_label),
                lambda: spec.peierls_vector_vertex(k, q, direction),
                lambda: spec.peierls_vector_vertex(k, q, direction_label),
            )
        ),
        dtype=complex,
    )


def _block_error_norms(error: np.ndarray, normal_dim: int | None) -> dict[str, float]:
    matrix = np.asarray(error, dtype=complex)
    if normal_dim is None or matrix.shape != (2 * normal_dim, 2 * normal_dim):
        return {"unknown": float(np.linalg.norm(matrix))}
    return {
        "particle_particle": float(np.linalg.norm(matrix[:normal_dim, :normal_dim])),
        "hole_hole": float(np.linalg.norm(matrix[normal_dim:, normal_dim:])),
        "pairing_offdiag": float(np.linalg.norm(matrix[:normal_dim, normal_dim:])),
        "conjugate_offdiag": float(np.linalg.norm(matrix[normal_dim:, :normal_dim])),
    }


def _worst_block(blocks: dict[str, float]) -> str:
    if not blocks:
        return "unknown"
    return max(blocks.items(), key=lambda item: item[1])[0]


def _operator_stats(errors: list[float], worst_k: list[float] | None) -> dict[str, Any]:
    if not errors:
        return {
            "max_error_norm": None,
            "rms_error_norm": None,
            "worst_k": None,
            "passed_by_tolerance": False,
        }
    return {
        "max_error_norm": float(max(errors)),
        "rms_error_norm": float(np.sqrt(np.mean(np.asarray(errors, dtype=float) ** 2))),
        "worst_k": worst_k,
        "passed_by_tolerance": bool(max(errors) <= _TOLERANCE),
    }


def _scaling_slope(q_values: list[float], residuals: list[float]) -> float | None:
    positive = [(q, r) for q, r in zip(q_values, residuals, strict=True) if q > 0.0 and r > 0.0]
    if len(positive) < 2:
        return None
    first_q, first_r = positive[0]
    last_q, last_r = positive[-1]
    if first_q == last_q:
        return None
    return float((np.log(last_r) - np.log(first_r)) / (np.log(last_q) - np.log(first_q)))


def _normal_vertex_identity(model: Any, points: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    errors_by_sign: dict[str, list[float]] = {"plus": [], "minus": []}
    worst_by_sign: dict[str, list[float] | None] = {"plus": None, "minus": None}
    for k in points:
        lhs = q[0] * _normal_vector_vertex(model.spec, k, q, 0) + q[1] * _normal_vector_vertex(model.spec, k, q, 1)
        rhs = _normal_hamiltonian(model.spec, k + 0.5 * q) - _normal_hamiltonian(model.spec, k - 0.5 * q)
        for sign_name, signed_rhs in (("plus", rhs), ("minus", -rhs)):
            error = float(np.linalg.norm(lhs - signed_rhs))
            errors_by_sign[sign_name].append(error)
            if error == max(errors_by_sign[sign_name]):
                worst_by_sign[sign_name] = [float(k[0]), float(k[1])]
    best_sign = min(errors_by_sign, key=lambda key: max(errors_by_sign[key]))
    stats = _operator_stats(errors_by_sign[best_sign], worst_by_sign[best_sign])
    stats.update(
        {
            "available": True,
            "best_sign_convention": best_sign,
            "signed_candidates": {
                sign: {
                    "max_error_norm": float(max(values)),
                    "rms_error_norm": float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2))),
                }
                for sign, values in errors_by_sign.items()
            },
            "interpretation": (
                "normal Peierls operator identity is within diagnostic tolerance"
                if stats["passed_by_tolerance"]
                else "normal Peierls operator identity mismatch is visible before response assembly"
            ),
        }
    )
    return stats


def _bdg_identity_unavailable(reason: str, pairings: tuple[str, ...]) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "by_pairing": {
            pairing: {
                "available": False,
                "reason": reason,
                "valid_for_casimir_input": False,
            }
            for pairing in pairings
        },
        "valid_for_casimir_input": False,
    }


def run_normal_finite_q_ward_triage(
    *,
    model_name: str = "symmetry_bdg_2band",
    q_model: tuple[float, float] = (0.01, 0.0),
    omega_eV: float = 0.01,
    nk: int = 9,
) -> dict[str, Any]:
    """Summarize normal-state finite-q Ward residuals without writing artifacts."""
    try:
        model = get_finite_q_validation_model(model_name)
        points = uniform_bz_mesh(nk)
        weights = k_weights(points)
        config = KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
        components = normal_physical_density_current_response_components_imag_axis_from_model(
            model.spec,
            points,
            config,
            np.asarray(q_model, dtype=float),
            weights,
        )
        bubble = _find_component_matrix(components, ("bare_bubble", "bubble", "paramagnetic"))
        direct = _find_component_matrix(components, ("direct", "contact", "diamagnetic"))
        total = _find_component_matrix(components, ("total", "bare_total"))
        if total is None:
            return _unavailable("normal finite-q response components did not include total or bare_total")
        if bubble is None or direct is None:
            residual_components = {
                "total": _summary_payload(_ward_residual_summary(total, omega_eV, q_model)),
            }
            suspected_layer = "unknown"
            interpretation = "normal finite-q API did not expose separate bubble/direct matrices"
        else:
            residual_components = {
                "bare_bubble": _summary_payload(_ward_residual_summary(bubble, omega_eV, q_model)),
                "direct": _summary_payload(_ward_residual_summary(direct, omega_eV, q_model)),
                "total": _summary_payload(_ward_residual_summary(total, omega_eV, q_model)),
            }
            total_norm = residual_components["total"]["max_norm"]
            bubble_norm = residual_components["bare_bubble"]["max_norm"]
            direct_norm = residual_components["direct"]["max_norm"]
            if total_norm <= _TOLERANCE:
                suspected_layer = "normal_closed"
            elif total_norm <= max(bubble_norm, direct_norm):
                suspected_layer = "normal_contact_or_vertex"
            else:
                suspected_layer = "normal_response_assembly"
            interpretation = "diagnostic-only normal finite-q Ward residual decomposition"
        total_summary = residual_components["total"]
        return {
            "available": True,
            "model_name": model.name,
            "q_model": [float(q_model[0]), float(q_model[1])],
            "omega_eV": float(omega_eV),
            "nk": int(nk),
            "workspace_evaluation": True,
            "valid_for_casimir_input": False,
            "residual_components": residual_components,
            "component_labels": list(COMPONENT_LABELS),
            "dominant_total_component": total_summary["dominant_component"],
            "normal_total_closed": bool(total_summary["max_norm"] <= _TOLERANCE),
            "suspected_layer": suspected_layer,
            "interpretation": interpretation,
        }
    except Exception as exc:  # diagnostic report should stay writable with an explicit reason
        return _unavailable(f"normal finite-q Ward triage failed: {type(exc).__name__}: {exc}")


def run_operator_identity_triage(
    *,
    model_name: str = "symmetry_bdg_2band",
    pairings: tuple[str, ...] = ("spm", "dwave"),
    q_model: tuple[float, float] = (0.01, 0.0),
    nk: int = 9,
    delta0_eV: float = 0.1,
) -> dict[str, Any]:
    """Check whether available operator-level identities fail before integration."""
    del delta0_eV
    try:
        model = get_finite_q_validation_model(model_name)
        points = uniform_bz_mesh(nk)
        q = np.asarray(q_model, dtype=float)
        try:
            normal_vertex = _normal_vertex_identity(model, points, q)
        except Exception as exc:
            normal_vertex = _unavailable(f"normal operator identity API unavailable: {type(exc).__name__}: {exc}")

        bdg_reason = "compatible BdG finite-q operator identity API was not found by triage helper"
        bdg_vertex = _bdg_identity_unavailable(bdg_reason, pairings)
        pairing_sector = _bdg_identity_unavailable(
            "compatible pairing-sector gauge variation API was not found by triage helper",
            pairings,
        )

        if normal_vertex.get("available") and not normal_vertex.get("passed_by_tolerance", False):
            suspected_layer = "normal_peierls_vertex"
        elif bdg_vertex.get("available") and any(
            not item.get("passed_by_tolerance", False) for item in bdg_vertex.get("by_pairing", {}).values()
        ):
            suspected_layer = "bdg_vertex_construction"
        elif pairing_sector.get("available") and any(
            not item.get("passed_by_tolerance", False) for item in pairing_sector.get("by_pairing", {}).values()
        ):
            suspected_layer = "pairing_gauge_vertex"
        elif normal_vertex.get("available") and normal_vertex.get("passed_by_tolerance", False):
            suspected_layer = "response_assembly_or_collective"
        else:
            suspected_layer = "unknown"

        return {
            "available": True,
            "model_name": model.name,
            "q_model": [float(q_model[0]), float(q_model[1])],
            "nk": int(nk),
            "valid_for_casimir_input": False,
            "normal_vertex": normal_vertex,
            "bdg_vertex": bdg_vertex,
            "pairing_sector": pairing_sector,
            "operator_identity_status": (
                "diagnostic_only"
                if not normal_vertex.get("available")
                else ("passed" if suspected_layer == "response_assembly_or_collective" else "failed")
            ),
            "suspected_layer": suspected_layer,
            "interpretation": "diagnostic-only operator identity triage; unavailable subchecks are explicit",
        }
    except Exception as exc:
        return _unavailable(f"operator identity triage failed: {type(exc).__name__}: {exc}")


def run_contact_cancellation_triage(
    *,
    model_name: str = "symmetry_bdg_2band",
    pairings: tuple[str, ...] = ("spm", "dwave"),
    q_model: tuple[float, float] = (0.01, 0.0),
    omega_eV: float = 0.01,
    nk: int = 9,
    delta0_eV: float = 0.1,
) -> dict[str, Any]:
    """Measure geometry of bubble/direct residual cancellation for existing scan rows."""
    try:
        from validation.scripts.bdg_finite_q.finite_q_ward_scan import run_finite_q_ward_scan

        model = get_finite_q_validation_model(model_name)
        scan = run_finite_q_ward_scan(
            pairings,
            model_name=model.name,
            omega_eV=omega_eV,
            q_values=(float(np.linalg.norm(np.asarray(q_model, dtype=float))),),
            q_directions=(q_model,),
            nk=nk,
            pairing_params=model.build_pairing_params(delta0_eV),
            q0_status={pairing: "diagnostic_only_not_rerun" for pairing in pairings},
        )
        by_pairing: dict[str, Any] = {}
        for pairing in pairings:
            rows = {row.response_name: row for row in scan.rows if row.pairing_name == pairing}
            if not {"bare_bubble", "direct", "bare_total"} <= set(rows):
                by_pairing[pairing] = _unavailable("finite-q scan rows did not include bare_bubble/direct/bare_total")
                continue
            bubble = _row_residual_vector(rows["bare_bubble"])
            direct = _row_residual_vector(rows["direct"])
            total = _row_residual_vector(rows["bare_total"])
            bubble_norm = float(np.linalg.norm(bubble))
            direct_norm = float(np.linalg.norm(direct))
            total_norm = float(np.linalg.norm(total))
            denom = max(bubble_norm * direct_norm, _EPS)
            cosine = float(np.real(np.vdot(bubble, -direct)) / denom)
            cancellation_fraction = float(1.0 - total_norm / max(bubble_norm + direct_norm, _EPS))
            if cancellation_fraction > 0.8 and cosine > 0.8:
                interpretation = "bubble_and_direct_cancel_well"
            elif cosine > 0.8:
                interpretation = "contact_magnitude_mismatch"
            elif bubble_norm > 2.0 * max(direct_norm, _EPS):
                interpretation = "bubble_dominant"
            elif direct_norm > 2.0 * max(bubble_norm, _EPS):
                interpretation = "direct_dominant"
            elif cosine < 0.2:
                interpretation = "contact_direction_mismatch"
            else:
                interpretation = "unknown"
            by_pairing[pairing] = {
                "available": True,
                "bubble_residual_norm": bubble_norm,
                "direct_residual_norm": direct_norm,
                "total_residual_norm": total_norm,
                "cosine_bubble_vs_minus_direct": cosine,
                "cancellation_fraction": cancellation_fraction,
                "dominant_remaining_component": _dominant_component(total),
                "interpretation": interpretation,
                "residual_vector_source": "left/right Ward residual vectors flattened from finite-q scan rows",
                "valid_for_casimir_input": False,
            }
        return {
            "available": True,
            "model_name": model.name,
            "q_model": [float(q_model[0]), float(q_model[1])],
            "omega_eV": float(omega_eV),
            "nk": int(nk),
            "by_pairing": by_pairing,
            "valid_for_casimir_input": False,
        }
    except Exception as exc:
        return _unavailable(f"contact cancellation triage failed: {type(exc).__name__}: {exc}")


def run_normal_contact_direct_audit(
    *,
    model_name: str = "symmetry_bdg_2band",
    q_model: tuple[float, float] = (0.01, 0.0),
    omega_eV: float = 0.01,
    nk: int = 9,
) -> dict[str, Any]:
    """Audit normal finite-q direct/contact placement and diagnostic-only residual candidates."""
    try:
        model = get_finite_q_validation_model(model_name)
        points = uniform_bz_mesh(nk)
        weights = k_weights(points)
        config = KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
        components = normal_physical_density_current_response_components_imag_axis_from_model(
            model.spec,
            points,
            config,
            np.asarray(q_model, dtype=float),
            weights,
        )
        bubble = _find_component_matrix(components, ("bare_bubble", "bubble", "paramagnetic"))
        direct = _find_component_matrix(components, ("direct", "contact", "diamagnetic"))
        total = _find_component_matrix(components, ("total", "bare_total"))
        if bubble is None or direct is None or total is None:
            missing = [
                name
                for name, value in (("bubble", bubble), ("direct", direct), ("total", total))
                if value is None
            ]
            return _unavailable(f"normal response components missing required matrices: {', '.join(missing)}")

        matrix_shape = list(total.shape)
        expected_shape_ok = bool(total.shape == (3, 3) and bubble.shape == total.shape and direct.shape == total.shape)
        direct_pattern = _direct_nonzero_pattern(direct)
        direct_block = _direct_block_interpretation(direct, expected_shape_ok)
        residual_audit = {
            "bare_bubble": _ward_residual_detail(bubble, omega_eV, q_model),
            "direct": _ward_residual_detail(direct, omega_eV, q_model),
            "total": _ward_residual_detail(total, omega_eV, q_model),
        }
        candidates = _direct_candidate_audit(bubble, direct, total, omega_eV, q_model)
        q_scaling = _normal_contact_q_scaling(model, omega_eV, nk)
        q0_convention = _q0_direct_convention_audit(model, omega_eV, nk)
        summary = _normal_contact_direct_summary(
            direct_block=direct_block,
            candidates=candidates,
            q_scaling=q_scaling,
            q0_convention=q0_convention,
        )
        return {
            "available": True,
            "model_name": model.name,
            "q_model": [float(q_model[0]), float(q_model[1])],
            "omega_eV": float(omega_eV),
            "nk": int(nk),
            "workspace_evaluation": True,
            "valid_for_casimir_input": False,
            "matrix_shape": matrix_shape,
            "assumed_component_labels": list(COMPONENT_LABELS),
            "label_to_index": {label: index for index, label in enumerate(COMPONENT_LABELS)},
            "is_square": bool(total.ndim == 2 and total.shape[0] == total.shape[1]),
            "expected_shape_ok": expected_shape_ok,
            "hermiticity_norms": {
                "bare_bubble": _hermiticity_norm(bubble),
                "direct": _hermiticity_norm(direct),
                "total": _hermiticity_norm(total),
            },
            "direct_nonzero_pattern": direct_pattern,
            "direct_block_interpretation": direct_block,
            "residual_component_audit": residual_audit,
            "direct_sign_candidates": candidates,
            "q_scaling": q_scaling,
            "q0_direct_convention": q0_convention,
            "summary": summary,
        }
    except Exception as exc:
        return _unavailable(f"normal contact/direct audit failed: {type(exc).__name__}: {exc}")


def _hermiticity_norm(matrix: np.ndarray) -> float:
    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return float("nan")
    return float(np.linalg.norm(array - array.conjugate().T))


def _direct_nonzero_pattern(direct: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(direct, dtype=complex)
    if matrix.shape != (3, 3):
        return {
            "density_density_norm": float("nan"),
            "density_current_norm": float("nan"),
            "current_density_norm": float("nan"),
            "current_current_norm": float("nan"),
            "current_x_current_x_norm": float("nan"),
            "current_x_current_y_norm": float("nan"),
            "current_y_current_x_norm": float("nan"),
            "current_y_current_y_norm": float("nan"),
        }
    return {
        "density_density_norm": float(abs(matrix[0, 0])),
        "density_current_norm": float(np.linalg.norm(matrix[0, 1:3])),
        "current_density_norm": float(np.linalg.norm(matrix[1:3, 0])),
        "current_current_norm": float(np.linalg.norm(matrix[1:3, 1:3])),
        "current_x_current_x_norm": float(abs(matrix[1, 1])),
        "current_x_current_y_norm": float(abs(matrix[1, 2])),
        "current_y_current_x_norm": float(abs(matrix[2, 1])),
        "current_y_current_y_norm": float(abs(matrix[2, 2])),
    }


def _direct_block_interpretation(direct: np.ndarray, expected_shape_ok: bool) -> str:
    if not expected_shape_ok:
        return "unexpected_shape"
    pattern = _direct_nonzero_pattern(direct)
    density_norm = max(
        pattern["density_density_norm"],
        pattern["density_current_norm"],
        pattern["current_density_norm"],
    )
    if density_norm > 1e-12:
        return "has_density_mixing"
    if pattern["current_current_norm"] > 0.0:
        return "current_current_only"
    return "unknown"


def _direct_candidate_audit(
    bubble: np.ndarray,
    direct: np.ndarray,
    total: np.ndarray,
    omega_eV: float,
    q_model: tuple[float, float],
) -> dict[str, Any]:
    current_block_direct = np.zeros_like(direct)
    xx_only_direct = np.zeros_like(direct)
    yy_only_direct = np.zeros_like(direct)
    if direct.shape == (3, 3):
        current_block_direct[1:3, 1:3] = direct[1:3, 1:3]
        xx_only_direct[1, 1] = direct[1, 1]
        yy_only_direct[2, 2] = direct[2, 2]
    candidates = {
        "total_current": total,
        "total_without_direct": bubble,
        "total_flip_direct": bubble - direct,
        "total_double_direct": bubble + 2.0 * direct,
        "total_half_direct": bubble + 0.5 * direct,
        "total_current_block_only_direct": bubble + current_block_direct,
        "total_xx_only_direct": bubble + xx_only_direct,
        "total_yy_only_direct": bubble + yy_only_direct,
    }
    current_summary = _ward_residual_summary(total, omega_eV, q_model)
    current_norm = max(current_summary.max_norm, _EPS)
    payload: dict[str, Any] = {}
    for name, matrix in candidates.items():
        summary = _ward_residual_summary(matrix, omega_eV, q_model)
        payload[name] = {
            "max_norm": summary.max_norm,
            "vector_norm": summary.norm,
            "dominant_component": summary.dominant_component,
            "improves_over_current_total": bool(summary.max_norm < current_summary.max_norm),
            "relative_to_current_total": float(summary.max_norm / current_norm),
        }
    return payload


def _normal_contact_q_scaling(model: Any, omega_eV: float, nk: int) -> dict[str, Any]:
    q_values = [0.005, 0.01, 0.02]
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    direct_residuals: list[float] = []
    total_residuals: list[float] = []
    for q_value in q_values:
        q_model = (q_value, 0.0)
        components = normal_physical_density_current_response_components_imag_axis_from_model(
            model.spec,
            points,
            config,
            np.asarray(q_model, dtype=float),
            weights,
        )
        direct = _find_component_matrix(components, ("direct", "contact", "diamagnetic"))
        total = _find_component_matrix(components, ("total", "bare_total"))
        if direct is None or total is None:
            return _unavailable("normal response components missing direct or total during q scaling audit")
        direct_residuals.append(_ward_residual_summary(direct, omega_eV, q_model).max_norm)
        total_residuals.append(_ward_residual_summary(total, omega_eV, q_model).max_norm)
    return {
        "available": True,
        "q_values": q_values,
        "direct_residuals": direct_residuals,
        "total_residuals": total_residuals,
        "direct_scaling_slope": _scaling_slope(q_values, direct_residuals),
        "total_scaling_slope": _scaling_slope(q_values, total_residuals),
        "valid_for_casimir_input": False,
    }


def _q0_direct_convention_audit(model: Any, omega_eV: float, nk: int) -> dict[str, Any]:
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(omega_eV=omega_eV, temperature_K=10.0, eta_eV=1e-8, output_si=False)
    components = normal_physical_density_current_response_components_imag_axis_from_model(
        model.spec,
        points,
        config,
        np.asarray((0.0, 0.0), dtype=float),
        weights,
    )
    direct = _find_component_matrix(components, ("direct", "contact", "diamagnetic"))
    if direct is None:
        return _unavailable("normal q=0 finite-q components did not expose direct/contact matrix")
    q0_direct_norm = float(np.linalg.norm(direct[1:3, 1:3] if direct.shape == (3, 3) else direct))
    return {
        "available": False,
        "reason": "local normal diamagnetic response-level comparator is not exposed by the current public API",
        "q0_direct_norm": q0_direct_norm,
        "q0_local_diamagnetic_norm": None,
        "best_q0_direct_match": "unavailable",
        "q0_direct_match_relative_error": None,
        "interpretation": "q=0 finite-q direct norm is recorded, but local diamagnetic sign relation was not compared",
        "valid_for_casimir_input": False,
    }


def _normal_contact_direct_summary(
    *,
    direct_block: str,
    candidates: dict[str, Any],
    q_scaling: dict[str, Any],
    q0_convention: dict[str, Any],
) -> dict[str, Any]:
    current = candidates.get("total_current", {})
    current_norm = float(current.get("max_norm", 0.0))
    improvement_threshold = 0.8 * max(current_norm, _EPS)
    flip_norm = float(candidates.get("total_flip_direct", {}).get("max_norm", float("inf")))
    half_norm = float(candidates.get("total_half_direct", {}).get("max_norm", float("inf")))
    double_norm = float(candidates.get("total_double_direct", {}).get("max_norm", float("inf")))
    current_block_norm = float(candidates.get("total_current_block_only_direct", {}).get("max_norm", float("inf")))
    xx_norm = float(candidates.get("total_xx_only_direct", {}).get("max_norm", float("inf")))
    yy_norm = float(candidates.get("total_yy_only_direct", {}).get("max_norm", float("inf")))

    if direct_block == "has_density_mixing":
        issue = "direct_has_density_mixing"
        fix = "Inspect why finite-q contact/direct contributes outside the current-current block."
    elif min(xx_norm, yy_norm, current_block_norm) < improvement_threshold:
        issue = "direct_component_placement_suspicious"
        fix = "Audit current-current component placement and density-current index ordering before changing formulas."
    elif flip_norm < improvement_threshold:
        issue = "direct_sign_suspicious"
        fix = "Audit finite-q direct/contact sign convention against the response-level Ward residual."
    elif min(half_norm, double_norm) < improvement_threshold:
        issue = "direct_magnitude_suspicious"
        fix = "Audit direct/contact normalization and any missing factor in response assembly."
    elif q0_convention.get("best_q0_direct_match") not in {None, "unavailable", "local_diamagnetic", "-local_diamagnetic"}:
        issue = "q0_finiteq_direct_convention_mismatch"
        fix = "Compare q=0 local direct convention with finite-q direct/contact convention."
    elif (
        q_scaling.get("available")
        and q_scaling.get("direct_scaling_slope") is not None
        and 0.5 <= float(q_scaling["direct_scaling_slope"]) <= 1.5
    ):
        issue = "q_factor_or_contact_scaling_suspicious"
        fix = "Audit q-factor placement in the contact/direct Ward contribution."
    elif direct_block == "unexpected_shape":
        issue = "ward_validator_component_order_suspicious"
        fix = "Audit matrix shape and component order expected by the Ward validator."
    else:
        issue = "normal_contact_unresolved"
        fix = "Use direct/contact block and candidate residual summaries to choose the next narrow audit."
    return {
        "suspected_issue": issue,
        "recommended_next_fix": fix,
        "valid_for_casimir_input": False,
    }


def _row_residual_vector(row: Any) -> np.ndarray:
    values: list[complex] = []
    for component in (*row.left_ward_residual_vector, *row.right_ward_residual_vector):
        values.append(complex(float(component["real"]), float(component["imag"])))
    return np.asarray(values, dtype=complex)


def summarize_ward_triage(
    normal_finite_q: dict[str, Any],
    operator_identity: dict[str, Any],
    contact_cancellation: dict[str, Any],
) -> dict[str, Any]:
    """Produce a compact diagnostic-only triage summary."""
    if operator_identity.get("suspected_layer") in {"normal_peierls_vertex", "bdg_vertex_construction", "pairing_gauge_vertex"}:
        suspected_layer = str(operator_identity["suspected_layer"])
        recommended = "Prioritize the operator-level identity indicated by suspected_layer before response assembly changes."
    elif normal_finite_q.get("suspected_layer") in {"normal_contact_or_vertex", "normal_response_assembly"}:
        suspected_layer = str(normal_finite_q["suspected_layer"])
        recommended = "Inspect normal finite-q contact/vertex assembly before collective-sector changes."
    else:
        pairings = contact_cancellation.get("by_pairing", {})
        interpretations = {
            item.get("interpretation")
            for item in pairings.values()
            if isinstance(item, dict) and item.get("available")
        }
        if "contact_direction_mismatch" in interpretations:
            suspected_layer = "contact_cancellation_geometry"
            recommended = "Inspect direct/contact residual direction relative to bubble residual."
        elif "contact_magnitude_mismatch" in interpretations:
            suspected_layer = "contact_cancellation_magnitude"
            recommended = "Inspect direct/contact residual normalization relative to bubble residual."
        else:
            suspected_layer = "response_assembly_or_collective"
            recommended = "If operator identities are acceptable, inspect response assembly and collective basis next."
    return {
        "suspected_layer": suspected_layer,
        "recommended_next_fix": recommended,
        "valid_for_casimir_input": False,
    }
