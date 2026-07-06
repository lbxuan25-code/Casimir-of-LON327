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
    return np.asarray(
        _call_first(
            (
                lambda: spec.current_vertex(direction, kx, ky, qx, qy),
                lambda: spec.current_vertex(kx, ky, qx, qy, direction),
                lambda: spec.current_vertex(k, q, direction),
                lambda: spec.vector_vertex(direction, kx, ky, qx, qy),
                lambda: spec.vector_vertex(kx, ky, qx, qy, direction),
                lambda: spec.vector_vertex(k, q, direction),
                lambda: spec.peierls_vector_vertex(direction, kx, ky, qx, qy),
                lambda: spec.peierls_vector_vertex(kx, ky, qx, qy, direction),
                lambda: spec.peierls_vector_vertex(k, q, direction),
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
