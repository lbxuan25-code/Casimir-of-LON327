"""Helpers for cached periodic-shift d-wave batch validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from lno327 import KuboConfig
from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.static_gauge_projection import (
    PROJECT_AFTER_VALIDATED_WARD,
    static_matsubara_kernel_to_sheet_response_with_policy,
)
from lno327.electrodynamics.static_sheet import (
    static_matsubara_kernel_to_sheet_response,
    static_sheet_response_to_reflection,
)
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.finite_q import BdGFiniteQResponseComponents
from lno327.response.ward_validation import PrimitiveWardRHS, validate_effective_ward_xy
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.dwave_periodic_shift_ensemble import (
    merge_shift_components_before_schur,
    periodic_shift_mesh,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


@dataclass(frozen=True)
class ShiftBatchConfig:
    base_nk: int
    qx: float
    qy: float
    temperature_K: float
    delta0_eV: float
    eta_eV: float
    ward_tolerance: float
    ward_absolute_tolerance: float
    condition_max: float
    raw_longitudinal_ceiling: float
    longitudinal_tolerance: float
    mixing_tolerance: float
    reality_tolerance: float
    passivity_tolerance: float
    separation_nm: float

    @property
    def q(self) -> np.ndarray:
        return np.asarray([self.qx, self.qy], dtype=float)


def evaluate_one_shift(config: ShiftBatchConfig, index: int, shift: np.ndarray) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("dwave", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(config.delta0_eV)
    kubo = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=config.temperature_K,
        eta_eV=config.eta_eV,
        output_si=False,
    )
    points, weights = periodic_shift_mesh(config.base_nk, shift)
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec, ansatz, points, weights, kubo, pairing, FiniteQEngineOptions()
    )
    workspace = precompute_finite_q_q_workspace(material, config.q)
    return {
        "index": int(index),
        "shift": np.asarray(shift, dtype=float),
        "components": finite_q_bdg_response_from_q_workspace(workspace, 0.0),
        "rhs": primitive_ward_rhs_from_q_workspace(workspace, 0.0),
        "workspace": workspace if int(index) == 0 else None,
    }


def _portable_component_payload(components: BdGFiniteQResponseComponents) -> dict[str, Any]:
    """Strip non-pickleable nested metadata from one worker result.

    ``BdGFiniteQResponseComponents.metadata`` may contain ``MappingProxyType``
    values inherited from immutable model metadata.  ProcessPool workers must
    therefore return a deliberately small plain-dict metadata contract.  Every
    numerical dataclass field is retained exactly; only unused audit metadata is
    omitted during inter-process transfer.
    """

    payload: dict[str, Any] = {}
    for field in fields(BdGFiniteQResponseComponents):
        if field.name == "metadata":
            payload[field.name] = {
                "phase_phase_direct_plus_convention": complex(
                    components.metadata["phase_phase_direct_plus_convention"]
                ),
                "phase_phase_direct_minus_convention": complex(
                    components.metadata["phase_phase_direct_minus_convention"]
                ),
                "worker_transfer": "portable_numeric_payload",
            }
        else:
            payload[field.name] = getattr(components, field.name)
    return payload


def portable_shift_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a ProcessPool-safe representation of ``evaluate_one_shift`` output."""

    rhs: PrimitiveWardRHS = result["rhs"]
    return {
        "index": int(result["index"]),
        "shift": np.asarray(result["shift"], dtype=float),
        "components": _portable_component_payload(result["components"]),
        "rhs": {
            "left": np.asarray(rhs.left, dtype=complex),
            "right": np.asarray(rhs.right, dtype=complex),
            "q_model": np.asarray(rhs.q_model, dtype=float),
            "xi_eV": float(rhs.xi_eV),
            "delta0_eV": float(rhs.delta0_eV),
            "metadata": {
                "convention": rhs.metadata.get("convention"),
                "basis": rhs.metadata.get("basis"),
                "source": "portable worker payload",
            },
        },
    }


def evaluate_one_shift_portable(
    config: ShiftBatchConfig, index: int, shift: np.ndarray
) -> dict[str, Any]:
    """Worker entry point returning only pickle-safe arrays, scalars and dicts."""

    return portable_shift_result(evaluate_one_shift(config, index, shift))


def restore_portable_shift_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct typed response objects in the parent process."""

    rhs_payload = dict(payload["rhs"])
    return {
        "index": int(payload["index"]),
        "shift": np.asarray(payload["shift"], dtype=float),
        "components": BdGFiniteQResponseComponents(**dict(payload["components"])),
        "rhs": PrimitiveWardRHS(**rhs_payload),
        "workspace": None,
    }


def _matrix_fields(matrix: np.ndarray) -> dict[str, float]:
    value = np.asarray(matrix, dtype=complex)
    result = {"reflection_norm": float(np.linalg.norm(value))}
    for label, i, j in (("ll", 0, 0), ("lt", 0, 1), ("tl", 1, 0), ("tt", 1, 1)):
        scalar = complex(value[i, j])
        result[f"reflection_{label}_real"] = float(scalar.real)
        result[f"reflection_{label}_imag"] = float(scalar.imag)
    return result


def postprocess_merged(components, rhs, config: ShiftBatchConfig) -> dict[str, Any]:
    kernel = effective_em_kernel_from_components(components, q_model=config.q, xi_eV=0.0)
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=config.ward_tolerance,
        absolute_residual_tolerance=config.ward_absolute_tolerance,
        condition_max=config.condition_max,
    )
    raw = static_matsubara_kernel_to_sheet_response(
        kernel,
        ward,
        longitudinal_tolerance=config.longitudinal_tolerance,
        mixing_tolerance=config.mixing_tolerance,
        reality_tolerance=config.reality_tolerance,
        passivity_tolerance=config.passivity_tolerance,
    )
    reflection_matrix = np.full((2, 2), np.nan + 1j * np.nan, dtype=complex)
    projection_error = ""
    reflection_error = ""
    logdet_error = ""
    projection_eligible = False
    reflection_constructed = False
    logdet_passed = False
    logdet = float("nan")
    try:
        projected = static_matsubara_kernel_to_sheet_response_with_policy(
            kernel,
            ward,
            longitudinal_policy=PROJECT_AFTER_VALIDATED_WARD,
            projection_raw_longitudinal_ceiling=config.raw_longitudinal_ceiling,
            longitudinal_tolerance=config.longitudinal_tolerance,
            mixing_tolerance=config.mixing_tolerance,
            reality_tolerance=config.reality_tolerance,
            passivity_tolerance=config.passivity_tolerance,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        projection_error = str(exc)
    else:
        projection_eligible = True
        try:
            reflection = static_sheet_response_to_reflection(
                projected, q_lab_model=config.q, theta_rad=0.0, require_physical=True
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            reflection_error = str(exc)
        else:
            reflection_constructed = True
            reflection_matrix = np.asarray(reflection.matrix_lt, dtype=complex)
            try:
                point = passive_sheet_logdet(
                    reflection,
                    reflection,
                    separation_m=config.separation_nm * 1e-9,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                logdet_error = str(exc)
            else:
                logdet_passed = True
                logdet = float(point.logdet)
    result: dict[str, Any] = {
        "ward_passed": bool(ward.passed),
        "ward_primitive_mixed_ratio_max": max(
            ward.left.primitive_mixed_ratio, ward.right.primitive_mixed_ratio
        ),
        "ward_effective_mixed_ratio_max": max(
            ward.left.effective_mixed_ratio, ward.right.effective_mixed_ratio
        ),
        "schur_condition_number": float(ward.schur_condition_number),
        "schur_inverse_method": ward.schur_inverse_method,
        "raw_longitudinal": float(raw.validation.relative_longitudinal_gauge_residual),
        "raw_imaginary": float(raw.validation.relative_imaginary_norm),
        "raw_density_transverse_mixing": float(
            raw.validation.relative_density_transverse_mixing
        ),
        "chi_bar": float(raw.chi_bar),
        "dbar_t": float(raw.dbar_t),
        "projection_eligible": projection_eligible,
        "projection_error": projection_error,
        "reflection_constructed": reflection_constructed,
        "reflection_error": reflection_error,
        "logdet_passed": logdet_passed,
        "logdet": logdet,
        "logdet_error": logdet_error,
    }
    result.update(_matrix_fields(reflection_matrix))
    return result


def merge_prefix(
    shift_results: list[dict[str, Any]],
    prefix: int,
    template_workspace,
    config: ShiftBatchConfig,
) -> dict[str, Any]:
    selected = shift_results[: int(prefix)]
    components, rhs = merge_shift_components_before_schur(
        [item["components"] for item in selected],
        [item["rhs"] for item in selected],
        np.ones(len(selected), dtype=float),
        template_workspace,
        omega_eV=0.0,
    )
    return postprocess_merged(components, rhs, config)


def jackknife_orbit_errors(
    shift_results: list[dict[str, Any]],
    prefix: int,
    template_workspace,
    config: ShiftBatchConfig,
) -> dict[str, float]:
    selected = shift_results[: int(prefix)]
    orbits = len(selected) // 4
    names = ("chi_bar", "dbar_t", "reflection_norm", "logdet")
    if orbits < 2:
        return {f"jackknife_{name}_abs": float("nan") for name in names}
    samples = {name: [] for name in names}
    for omitted in range(orbits):
        retained = [item for index, item in enumerate(selected) if index // 4 != omitted]
        components, rhs = merge_shift_components_before_schur(
            [item["components"] for item in retained],
            [item["rhs"] for item in retained],
            np.ones(len(retained), dtype=float),
            template_workspace,
            omega_eV=0.0,
        )
        result = postprocess_merged(components, rhs, config)
        for name in names:
            samples[name].append(float(result[name]))
    errors: dict[str, float] = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            errors[f"jackknife_{name}_abs"] = float("nan")
            continue
        mean = float(np.mean(array))
        errors[f"jackknife_{name}_abs"] = float(
            np.sqrt((orbits - 1.0) / orbits * np.sum((array - mean) ** 2))
        )
    return errors
