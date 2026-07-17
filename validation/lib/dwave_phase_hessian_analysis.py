"""Post-process a commensurate d-wave Ward audit without rerunning the BZ integral.

The expensive commensurate runner stores the component-resolved collective-column
identity in JSON.  This module compares the observed phase-column cancellation with
three scalar counterterm hypotheses:

1. the current q-independent Goldstone counterterm;
2. the nearest-neighbour bond metric
   ``(cos(qx/2)^2 + cos(qy/2)^2) / 2``;
3. the curvature inferred from the already-integrated phase-angle direct term.

The same parser also accepts the reduced phase-column-only commensurate payload,
which preserves exactly the collective pieces needed by this analysis while omitting
unrelated electromagnetic and Ward channels.  All outputs remain diagnostic only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

_ACCEPTED_SCHEMAS = {
    "dwave_static_commensurate_periodic_ward_audit_v1",
    "dwave_static_commensurate_phase_column_audit_v1",
}


@dataclass(frozen=True)
class PhaseHessianSideAnalysis:
    orientation: str
    em_collective_phase: complex
    phase_rotation_bubble: complex
    phase_rotation_counterterm: complex
    current_phase_defect: complex
    required_counterterm_multiplier: complex
    bond_metric_phase_defect: complex
    phase_direct_phase_defect: complex | None


@dataclass(frozen=True)
class DWavePhaseHessianAnalysis:
    q_model: tuple[float, float]
    q_norm: float
    delta0_eV: float
    bond_metric_multiplier: float
    phase_direct_plus: complex | None
    phase_direct_curvature: complex | None
    counterterm_curvature: complex
    phase_direct_counterterm_multiplier: complex | None
    left: PhaseHessianSideAnalysis
    right: PhaseHessianSideAnalysis
    diagnostic_only: bool = True
    projection_applied: bool = False
    production_reference_established: bool = False
    valid_for_casimir_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _complex_scalar(value: Any, name: str) -> complex:
    if isinstance(value, Mapping) and set(value) >= {"real", "imag"}:
        scalar = complex(float(value["real"]), float(value["imag"]))
    else:
        scalar = complex(value)
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise ValueError(f"{name} must be finite")
    return scalar


def _complex_vector(value: Any, length: int, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} entries")
    vector = np.asarray(
        [_complex_scalar(item, f"{name}[{index}]") for index, item in enumerate(value)],
        dtype=complex,
    )
    return vector


def _real_vector(value: Any, length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    return vector


def _phase_component(
    audit: Mapping[str, Any], orientation: str, source_name: str
) -> complex:
    try:
        source = audit["component_sources"][orientation]["collective_defect_parts"][
            source_name
        ]
    except KeyError as exc:
        raise ValueError(
            f"audit is missing {orientation} collective source {source_name!r}"
        ) from exc
    return complex(_complex_vector(source, 2, source_name)[1])


def _safe_multiplier(numerator: complex, denominator: complex, name: str) -> complex:
    if abs(denominator) <= 1e-30:
        raise ValueError(f"cannot infer {name}: denominator is at the absolute floor")
    return complex(numerator / denominator)


def _side_analysis(
    audit: Mapping[str, Any],
    orientation: str,
    *,
    bond_metric: float,
    phase_direct_multiplier: complex | None,
) -> PhaseHessianSideAnalysis:
    em = _phase_component(audit, orientation, "em_collective_contraction")
    bubble = _phase_component(audit, orientation, "phase_rotation_bubble")
    counterterm = _phase_component(audit, orientation, "phase_rotation_counterterm")
    required = _safe_multiplier(-(em + bubble), counterterm, "counterterm multiplier")
    current = em + bubble + counterterm
    bond_candidate = em + bubble + bond_metric * counterterm
    direct_candidate = (
        None
        if phase_direct_multiplier is None
        else em + bubble + phase_direct_multiplier * counterterm
    )
    return PhaseHessianSideAnalysis(
        orientation=orientation,
        em_collective_phase=em,
        phase_rotation_bubble=bubble,
        phase_rotation_counterterm=counterterm,
        current_phase_defect=current,
        required_counterterm_multiplier=required,
        bond_metric_phase_defect=bond_candidate,
        phase_direct_phase_defect=direct_candidate,
    )


def analyze_dwave_phase_hessian_payload(
    payload: Mapping[str, Any],
) -> DWavePhaseHessianAnalysis:
    """Analyze one full or reduced commensurate d-wave phase-column payload."""

    schema = payload.get("schema")
    if schema not in _ACCEPTED_SCHEMAS:
        choices = ", ".join(sorted(_ACCEPTED_SCHEMAS))
        raise ValueError(f"input schema must be one of: {choices}")
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("payload is missing the audit mapping")

    q = _real_vector(audit.get("q_model"), 2, "audit.q_model")
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        raise ValueError("commensurate phase-Hessian analysis requires nonzero q")
    delta0 = float(audit.get("delta0_eV"))
    if not np.isfinite(delta0) or delta0 <= 0.0:
        raise ValueError("audit.delta0_eV must be finite and positive")

    bond_metric = float(
        0.5 * (np.cos(0.5 * q[0]) ** 2 + np.cos(0.5 * q[1]) ** 2)
    )

    w_left = _complex_vector(audit.get("w_left"), 2, "audit.w_left")
    counterterm_phase_left = _phase_component(
        audit, "left", "phase_rotation_counterterm"
    )
    counterterm_curvature = _safe_multiplier(
        counterterm_phase_left, complex(w_left[1]), "counterterm curvature"
    )

    primitive_metadata = payload.get("primitive_metadata", {})
    phase_direct_plus: complex | None = None
    phase_direct_curvature: complex | None = None
    phase_direct_multiplier: complex | None = None
    if isinstance(primitive_metadata, Mapping) and "phase_phase_direct_plus" in primitive_metadata:
        phase_direct_plus = _complex_scalar(
            primitive_metadata["phase_phase_direct_plus"],
            "primitive_metadata.phase_phase_direct_plus",
        )
        phase_direct_curvature = phase_direct_plus / (delta0 * delta0)
        phase_direct_multiplier = _safe_multiplier(
            phase_direct_curvature,
            counterterm_curvature,
            "phase-direct counterterm multiplier",
        )

    left = _side_analysis(
        audit,
        "left",
        bond_metric=bond_metric,
        phase_direct_multiplier=phase_direct_multiplier,
    )
    right = _side_analysis(
        audit,
        "right",
        bond_metric=bond_metric,
        phase_direct_multiplier=phase_direct_multiplier,
    )
    return DWavePhaseHessianAnalysis(
        q_model=(float(q[0]), float(q[1])),
        q_norm=q_norm,
        delta0_eV=delta0,
        bond_metric_multiplier=bond_metric,
        phase_direct_plus=phase_direct_plus,
        phase_direct_curvature=phase_direct_curvature,
        counterterm_curvature=counterterm_curvature,
        phase_direct_counterterm_multiplier=phase_direct_multiplier,
        left=left,
        right=right,
    )


def complex_jsonable(value: Any) -> Any:
    """Convert dataclass output containing complex values into JSON-safe objects."""

    if isinstance(value, complex | np.complexfloating):
        scalar = complex(value)
        return {"real": float(scalar.real), "imag": float(scalar.imag)}
    if isinstance(value, np.generic):
        return complex_jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): complex_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [complex_jsonable(item) for item in value]
    return value


__all__ = [
    "DWavePhaseHessianAnalysis",
    "PhaseHessianSideAnalysis",
    "analyze_dwave_phase_hessian_payload",
    "complex_jsonable",
]
