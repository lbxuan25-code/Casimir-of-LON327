"""Blocking commensurate/arbitrary-q qualification for the periodic-BZ backend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np

from lno327 import KuboConfig
from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.electrodynamics.conventions import (
    positive_matsubara_kernel_to_sheet_response,
    validate_positive_matsubara_sheet_response,
)
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE
from lno327.electrodynamics.reflection import positive_matsubara_sheet_response_to_reflection
from lno327.electrodynamics.static_sheet import (
    static_matsubara_kernel_to_sheet_response,
    static_sheet_response_to_reflection,
)
from lno327.response.arbitrary_q_material_cache import build_material_grid_cache
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.periodic_bz_grid import build_periodic_bz_grid
from lno327.response.static_ward_gate import validate_strict_static_ward_closure
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.workflows.arbitrary_q_matsubara import (
    integrate_arbitrary_q_periodic_bz,
    integrate_two_plate_angle_batch,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.commands.matsubara.positive_point import matsubara_energy_eV
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
)
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.positive_orbit_gauss import integrate_matsubara_orbit_gauss

DEFAULT_OUTPUT = Path("validation/outputs/matsubara/arbitrary_q_periodic_bz_qualification/qualification.json")


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairings", nargs="+", choices=("spm", "dwave"), default=["spm", "dwave"])
    p.add_argument("--N-values", nargs="+", type=int, default=[256, 384, 512])
    p.add_argument("--reference-nk", type=int, default=1256)
    p.add_argument("--reference-order", type=int, default=384)
    p.add_argument("--reference-panel-count", type=int, default=16)
    p.add_argument("--reference-workers", type=int, default=8)
    p.add_argument("--reference-task-size", type=int, default=4)
    p.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1, 8])
    p.add_argument("--canonical-block-size", type=int, default=4096)
    p.add_argument("--runtime-chunk-size", type=int, default=16384)
    p.add_argument("--temperature-K", type=float, default=10.0)
    p.add_argument("--delta0-eV", type=float, default=0.1)
    p.add_argument("--eta-eV", type=float, default=1e-8)
    p.add_argument("--separation-nm", type=float, default=20.0)
    p.add_argument("--primitive-tolerance", type=float, default=1e-3)
    p.add_argument("--reflection-tolerance", type=float, default=3e-4)
    p.add_argument("--logdet-tolerance", type=float, default=3e-4)
    p.add_argument("--diagonal-observable-tolerance", type=float, default=1e-3)
    p.add_argument("--ward-tolerance", type=float, default=1e-7)
    p.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args(argv)
    a.N_values = tuple(sorted(set(a.N_values)))
    a.matsubara_indices = tuple(sorted(set(a.matsubara_indices)))
    a.pairings = tuple(dict.fromkeys(a.pairings))
    if len(a.N_values) < 3 or any(n <= 0 or n % 2 for n in a.N_values):
        p.error("at least three positive even N values are required")
    if 0 not in a.matsubara_indices or not any(n > 0 for n in a.matsubara_indices):
        p.error("exact zero and at least one positive Matsubara index are required")
    if a.reference_order % a.reference_panel_count:
        p.error("reference order must be divisible by panel count")
    return a


def _head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _xi(a: argparse.Namespace) -> np.ndarray:
    return np.asarray([0.0 if n == 0 else matsubara_energy_eV(n, a.temperature_K) for n in a.matsubara_indices])


def _relative(left: Any, right: Any, tolerance: float) -> dict[str, Any]:
    x, y = np.asarray(left, dtype=complex), np.asarray(right, dtype=complex)
    delta = float(np.linalg.norm(x - y))
    scale = max(float(np.linalg.norm(x)), float(np.linalg.norm(y)), np.finfo(float).tiny)
    value = delta / scale
    return {"relative": value, "tolerance": float(tolerance), "passed": bool(np.isfinite(value) and value <= tolerance)}


def _signature(component: object, rhs: object) -> np.ndarray:
    return np.concatenate([np.asarray(v, dtype=complex).reshape(-1) for v in (
        component.bare_bubble, component.direct, component.collective_bubble,
        component.collective_counterterm, component.em_collective_left,
        component.collective_em_right, component.gauge_restored, rhs.left, rhs.right,
    )])


def _physical(result: object, q: np.ndarray, a: argparse.Namespace) -> list[dict[str, Any]]:
    config = OrbitAcceptancePhysicsConfig(
        separation_nm=a.separation_nm,
        ward_tolerance=a.ward_tolerance,
        ward_absolute_tolerance=a.ward_absolute_tolerance,
    )
    rows = []
    for n, frequency, component, rhs in zip(a.matsubara_indices, result.xi_eV_values, result.components, result.rhs, strict=True):
        state = evaluate_matsubara_pipeline(components=component, rhs=rhs, q_model=q, xi_eV=float(frequency), config=config)
        rows.append({
            "n": int(n), "passed": bool(state["physical_passed"]),
            "ward": bool(state["ward_passed"]), "static": bool(state["strict_static_ward_passed"]),
            "reflection": np.asarray(state["reflection"], dtype=complex),
            "logdet": float(state["logdet"]), "primary": np.asarray(state["primary_response"], dtype=complex),
            "error": str(state["error"]),
        })
    return rows


def _periodic(a: argparse.Namespace, model: object, ansatz: object, pairing: object, q: np.ndarray, n: int, shift=(0.5, 0.5)):
    return integrate_arbitrary_q_periodic_bz(
        spec=model.spec, ansatz=ansatz, pairing=pairing, xi_eV_values=_xi(a),
        temperature_K=a.temperature_K, eta_eV=a.eta_eV, q_model=q, n=n, shift=shift,
        canonical_reduction_block_size=a.canonical_block_size,
        runtime_chunk_size=a.runtime_chunk_size,
    )


def _reference(a: argparse.Namespace, model: object, ansatz: object, pairing: object, indices: tuple[int, int]):
    mx, my = indices
    origins = 2 if int(np.gcd(abs(mx), abs(my))) % 2 else 1
    return integrate_matsubara_orbit_gauss(
        spec=model.spec, ansatz=ansatz, pairing=pairing, xi_eV_values=_xi(a),
        temperature_K=a.temperature_K, eta_eV=a.eta_eV, nk=a.reference_nk, mx=mx, my=my,
        transverse_order=a.reference_order, panel_count=a.reference_panel_count,
        shift_s=0.5, subgrid_average="auto",
        max_point_evaluations=a.reference_nk * origins * a.reference_order,
        transverse_workers=a.reference_workers, transverse_task_size=a.reference_task_size,
    )


def _commensurate(a: argparse.Namespace, pairing_name: str, name: str, indices: tuple[int, int]) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(a.delta0_eV)
    q = 2.0 * np.pi / a.reference_nk * np.asarray(indices, dtype=float)
    reference = _reference(a, model, ansatz, pairing, indices)
    reference_states = _physical(reference, q, a)
    series = [_periodic(a, model, ansatz, pairing, q, n) for n in a.N_values]
    states = [_physical(result, q, a) for result in series]
    diagonal = pairing_name == "dwave" and name == "exact_diagonal"
    observable_tolerance = a.diagonal_observable_tolerance if diagonal else a.reflection_tolerance
    frequency_rows = []
    for i, n in enumerate(a.matsubara_indices):
        primitive = _relative(_signature(reference.components[i], reference.rhs[i]), _signature(series[-1].components[i], series[-1].rhs[i]), a.primitive_tolerance)
        reflection = _relative(reference_states[i]["reflection"], states[-1][i]["reflection"], observable_tolerance)
        logdet = _relative(reference_states[i]["logdet"], states[-1][i]["logdet"], a.diagonal_observable_tolerance if diagonal else a.logdet_tolerance)
        refinement = [_relative(states[j][i]["reflection"], states[j + 1][i]["reflection"], observable_tolerance) for j in range(len(states) - 1)]
        logdet_refinement = [_relative(states[j][i]["logdet"], states[j + 1][i]["logdet"], a.diagonal_observable_tolerance if diagonal else a.logdet_tolerance) for j in range(len(states) - 1)]
        passed = bool(series[-1].operator_ward.passed and states[-1][i]["passed"] and reflection["passed"] and logdet["passed"] and all(x["passed"] for x in refinement + logdet_refinement) and (primitive["passed"] or diagonal))
        frequency_rows.append({"n": int(n), "primitive": primitive, "reflection": reflection, "logdet": logdet, "refinement": refinement, "logdet_refinement": logdet_refinement, "response_unresolved_allowed": diagonal, "passed": passed})
    return {"pairing": pairing_name, "case": name, "q_indices": list(indices), "q_model": q.tolist(), "N_values": list(a.N_values), "operator_ward": series[-1].operator_ward.as_dict(), "frequencies": frequency_rows, "passed": all(row["passed"] for row in frequency_rows)}


def _arbitrary(a: argparse.Namespace, pairing_name: str) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(a.delta0_eV)
    base = 2.0 * np.pi / a.reference_nk * np.asarray([6.0, 4.0])
    angle = np.deg2rad(17.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    q = rotation @ base
    series = [_periodic(a, model, ansatz, pairing, q, n) for n in a.N_values]
    states = [_physical(result, q, a) for result in series]
    audits = [_periodic(a, model, ansatz, pairing, q, a.N_values[-1], shift) for shift in ((0.25, 0.75), (0.75, 0.25))]
    audit_states = [_physical(result, q, a) for result in audits]
    rows = []
    for i, n in enumerate(a.matsubara_indices):
        primary = states[-1][i]
        response_refinement = [_relative(states[j][i]["primary"], states[j + 1][i]["primary"], a.primitive_tolerance) for j in range(len(states) - 1)]
        reflection_refinement = [_relative(states[j][i]["reflection"], states[j + 1][i]["reflection"], a.reflection_tolerance) for j in range(len(states) - 1)]
        logdet_refinement = [_relative(states[j][i]["logdet"], states[j + 1][i]["logdet"], a.logdet_tolerance) for j in range(len(states) - 1)]
        audit_response = 0.5 * (audit_states[0][i]["primary"] + audit_states[1][i]["primary"])
        audit_reflection = 0.5 * (audit_states[0][i]["reflection"] + audit_states[1][i]["reflection"])
        audit_logdet = 0.5 * (audit_states[0][i]["logdet"] + audit_states[1][i]["logdet"])
        shift = {
            "response": _relative(primary["primary"], audit_response, a.primitive_tolerance),
            "reflection": _relative(primary["reflection"], audit_reflection, a.reflection_tolerance),
            "logdet": _relative(primary["logdet"], audit_logdet, a.logdet_tolerance),
        }
        passed = bool(series[-1].operator_ward.passed and primary["passed"] and all(x["passed"] for x in response_refinement + reflection_refinement + logdet_refinement) and all(x["passed"] for x in shift.values()))
        rows.append({"n": int(n), "response_refinement": response_refinement, "reflection_refinement": reflection_refinement, "logdet_refinement": logdet_refinement, "shift_audit": shift, "passed": passed})
    return {"pairing": pairing_name, "case": "rotated_17deg", "q_model": q.tolist(), "operator_ward": series[-1].operator_ward.as_dict(), "frequencies": rows, "passed": all(row["passed"] for row in rows)}


def _plate_reflection(component: object, rhs: object, q_crystal: np.ndarray, q_lab: np.ndarray, theta: float, frequency: float, config: OrbitAcceptancePhysicsConfig):
    kernel = effective_em_kernel_from_components(component, q_model=q_crystal, xi_eV=frequency)
    ward = validate_effective_ward_xy(kernel, rhs, residual_tolerance=config.ward_tolerance, absolute_residual_tolerance=config.ward_absolute_tolerance, condition_max=config.condition_max)
    if frequency == 0.0:
        strict = validate_strict_static_ward_closure(kernel, ward, energy_scale_eV=config.static_energy_scale_eV, primitive_tolerance=config.static_primitive_tolerance, amplitude_tolerance=config.static_amplitude_tolerance, phase_tolerance=config.static_phase_tolerance, effective_direct_tolerance=config.static_effective_direct_tolerance, effective_residual_tolerance=config.static_effective_residual_tolerance, longitudinal_tolerance=config.static_longitudinal_tolerance, condition_max=config.condition_max)
        sheet = static_matsubara_kernel_to_sheet_response(kernel, ward, energy_scale_eV=config.static_energy_scale_eV, degeneracy=config.degeneracy, reality_tolerance=config.static_reality_tolerance, longitudinal_tolerance=config.static_longitudinal_tolerance, mixing_tolerance=config.static_mixing_tolerance, passivity_tolerance=config.static_passivity_tolerance)
        reflection = static_sheet_response_to_reflection(sheet, q_lab_model=q_lab, theta_rad=theta, lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m, require_physical=True)
        return reflection, bool(ward.passed and strict.passed and sheet.validation.passed)
    sheet = positive_matsubara_kernel_to_sheet_response(kernel, degeneracy=config.degeneracy)
    validation = validate_positive_matsubara_sheet_response(sheet)
    reflection = positive_matsubara_sheet_response_to_reflection(sheet, q_lab_model=q_lab, theta_rad=theta, lattice_constant_m=LNO327_THIN_FILM_SLAO_IN_PLANE.lattice_a_x_m, require_physical=True)
    return reflection, bool(ward.passed and validation.passed)


def _two_plate(a: argparse.Namespace, pairing_name: str) -> dict[str, Any]:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(a.delta0_eV)
    xi = _xi(a)
    q_lab = 2.0 * np.pi / a.reference_nk * np.asarray([6.0, 4.0])
    grid = build_periodic_bz_grid(a.N_values[-1], (0.5, 0.5))
    cache = build_material_grid_cache(spec=model.spec, ansatz=ansatz, pairing=pairing, config=KuboConfig.from_kelvin(omega_eV=float(xi[0]), temperature_K=a.temperature_K, eta_eV=a.eta_eV, output_si=False), options=FiniteQEngineOptions(phase_hessian_policy="q_independent"), grid=grid)
    theta = np.deg2rad(17.0)
    batch = integrate_two_plate_angle_batch(q_lab=q_lab, theta_1_rad=0.0, theta_2_rad_values=np.asarray([theta]), material_cache=cache, spec=model.spec, ansatz=ansatz, pairing=pairing, xi_eV_values=xi, temperature_K=a.temperature_K, eta_eV=a.eta_eV, canonical_reduction_block_size=a.canonical_block_size, runtime_chunk_size=a.runtime_chunk_size)
    config = OrbitAcceptancePhysicsConfig(separation_nm=a.separation_nm, ward_tolerance=a.ward_tolerance, ward_absolute_tolerance=a.ward_absolute_tolerance)
    rows = []
    for i, frequency in enumerate(xi):
        try:
            r1, p1 = _plate_reflection(batch.plate_1.components[i], batch.plate_1.rhs[i], batch.plate_1.q_model, q_lab, 0.0, float(frequency), config)
            r2, p2 = _plate_reflection(batch.plate_2[0].components[i], batch.plate_2[0].rhs[i], batch.plate_2[0].q_model, q_lab, theta, float(frequency), config)
            point = passive_sheet_logdet(r1, r2, separation_m=a.separation_nm * 1e-9)
            passed, error, logdet = bool(p1 and p2 and np.isfinite(point.logdet)), "", float(point.logdet)
        except (ValueError, np.linalg.LinAlgError) as exc:
            passed, error, logdet = False, str(exc), float("nan")
        rows.append({"n": int(a.matsubara_indices[i]), "logdet": logdet, "passed": passed, "error": error})
    return {"pairing": pairing_name, "q_lab": q_lab.tolist(), "theta_deg": [0.0, 17.0], "q_crystal": [batch.plate_1.q_model.tolist(), batch.plate_2[0].q_model.tolist()], "response_cache": batch.response_cache_metadata, "frequencies": rows, "passed": all(row["passed"] for row in rows)}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    a = _args(argv)
    cases = {"axis": (1, 0), "generic": (6, 4), "near_diagonal": (25, 24), "exact_diagonal": (6, 6)}
    commensurate = [_commensurate(a, pairing, name, indices) for pairing in a.pairings for name, indices in cases.items()]
    arbitrary = [_arbitrary(a, pairing) for pairing in a.pairings]
    two_plate = [_two_plate(a, pairing) for pairing in a.pairings]
    passed = all(row["passed"] for row in (*commensurate, *arbitrary, *two_plate))
    payload = {
        "schema": "arbitrary-q-periodic-bz-qualification-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": _head(),
        "config": {"pairings": list(a.pairings), "N_values": list(a.N_values), "reference_nk": a.reference_nk, "reference_order": a.reference_order, "matsubara_indices": list(a.matsubara_indices)},
        "commensurate_regression": commensurate, "arbitrary_q_refinement_and_shift_audit": arbitrary, "two_plate_common_lab_basis": two_plate,
        "arbitrary_q_microscopic_contract": "qualified_for_diagnostic_outer_integration" if passed else "qualification_failed",
        "diagnostic_only": True, "production_reference_established": False, "valid_for_casimir_input": False, "passed": passed,
    }
    _write(a.output, payload)
    print(json.dumps({"output": str(a.output), "passed": passed}, indent=2))
    if not passed:
        raise SystemExit("arbitrary-q periodic BZ qualification failed")


if __name__ == "__main__":
    main()
