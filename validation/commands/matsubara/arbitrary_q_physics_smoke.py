"""Small nonformal arbitrary-q physical-closure smoke test without convergence claims."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.static_ward_gate import validate_strict_static_ward_closure
from lno327.response.ward_validation import validate_effective_ward_xy
from validation.commands.matsubara import arbitrary_q_periodic_bz_qualification as qualification
from validation.lib.dwave_orbit_acceptance import (
    OrbitAcceptancePhysicsConfig,
    evaluate_matsubara_pipeline,
)
from validation.lib.source_tree_provenance import source_tree_provenance

DEFAULT_OUTPUT = Path(
    "validation/outputs/matsubara/arbitrary_q_staged_flow/stage2_physics_smoke.json"
)
_STATIC_METRICS = (
    "primitive_residual_over_q",
    "amplitude_defect_over_q",
    "phase_defect_over_q",
    "effective_direct_over_q",
    "effective_residual_over_q",
    "relative_longitudinal_gauge_residual",
)


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairings", nargs="+", choices=("spm", "dwave"), default=["spm", "dwave"])
    parser.add_argument("--N-values", nargs="+", type=int, default=[128, 192])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reference-nk", type=int, default=1256)
    parser.add_argument("--matsubara-indices", nargs="+", type=int, default=[0, 1, 8])
    parser.add_argument("--canonical-block-size", type=int, default=4096)
    parser.add_argument("--runtime-chunk-size", type=int, default=16384)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.pairings = tuple(dict.fromkeys(args.pairings))
    args.N_values = tuple(sorted(set(int(v) for v in args.N_values)))
    args.matsubara_indices = tuple(sorted(set(int(v) for v in args.matsubara_indices)))
    if any(v <= 0 or v % 2 for v in args.N_values):
        parser.error("positive even N values are required")
    if args.reference_nk <= 0:
        parser.error("reference nk must be positive")
    if args.workers <= 0:
        parser.error("workers must be positive")
    if 0 not in args.matsubara_indices or not any(v > 0 for v in args.matsubara_indices):
        parser.error("exact zero and at least one positive Matsubara index are required")
    if args.runtime_chunk_size < args.canonical_block_size:
        parser.error("runtime chunk must be at least the canonical block")
    if args.runtime_chunk_size % args.canonical_block_size:
        parser.error("runtime chunk must be a multiple of the canonical block")
    return args


def _physics_config(args: argparse.Namespace) -> OrbitAcceptancePhysicsConfig:
    return OrbitAcceptancePhysicsConfig(
        separation_nm=float(args.separation_nm),
        ward_tolerance=float(args.ward_tolerance),
        ward_absolute_tolerance=float(args.ward_absolute_tolerance),
    )


def _state(
    *,
    component: object,
    rhs: object,
    q_model: np.ndarray,
    xi_eV: float,
    config: OrbitAcceptancePhysicsConfig,
) -> dict[str, Any]:
    kernel = effective_em_kernel_from_components(
        component,
        q_model=np.asarray(q_model, dtype=float),
        xi_eV=float(xi_eV),
    )
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=config.ward_tolerance,
        absolute_residual_tolerance=config.ward_absolute_tolerance,
        condition_max=config.condition_max,
    )
    strict: dict[str, Any] | None = None
    if float(xi_eV) == 0.0:
        strict = validate_strict_static_ward_closure(
            kernel,
            ward,
            energy_scale_eV=config.static_energy_scale_eV,
            primitive_tolerance=config.static_primitive_tolerance,
            amplitude_tolerance=config.static_amplitude_tolerance,
            phase_tolerance=config.static_phase_tolerance,
            effective_direct_tolerance=config.static_effective_direct_tolerance,
            effective_residual_tolerance=config.static_effective_residual_tolerance,
            longitudinal_tolerance=config.static_longitudinal_tolerance,
            condition_max=config.condition_max,
        ).to_dict()
    pipeline = evaluate_matsubara_pipeline(
        components=component,
        rhs=rhs,
        q_model=np.asarray(q_model, dtype=float),
        xi_eV=float(xi_eV),
        config=config,
    )
    ward_ratio = max(ward.left.effective_mixed_ratio, ward.right.effective_mixed_ratio)
    return {
        "xi_eV": float(xi_eV),
        "response_sector": str(pipeline["response_sector"]),
        "integrated_ward_passed": bool(ward.passed),
        "ward_effective_mixed_ratio_max": float(ward_ratio),
        "schur_condition_number": float(ward.schur_condition_number),
        "strict_static": strict,
        "pipeline": {
            "physical_passed": bool(pipeline["physical_passed"]),
            "ward_passed": bool(pipeline["ward_passed"]),
            "strict_static_ward_passed": bool(pipeline["strict_static_ward_passed"]),
            "sheet_validation_passed": bool(pipeline["sheet_validation_passed"]),
            "reflection_constructed": bool(pipeline["reflection_constructed"]),
            "logdet_passed": bool(pipeline["logdet_passed"]),
            "logdet": float(pipeline["logdet"]),
            "error": str(pipeline["error"]),
        },
    }


def _response_states(args: argparse.Namespace, response: object) -> list[dict[str, Any]]:
    config = _physics_config(args)
    return [
        {
            "n": int(n_value),
            **_state(
                component=component,
                rhs=rhs,
                q_model=response.q_model,
                xi_eV=float(xi),
                config=config,
            ),
        }
        for n_value, xi, component, rhs in zip(
            args.matsubara_indices,
            response.xi_eV_values,
            response.components,
            response.rhs,
            strict=True,
        )
    ]


def _trend(rows_by_n: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows_by_n:
        return {}
    result: dict[str, Any] = {}
    for metric in _STATIC_METRICS:
        values = [float(row["strict_static"][metric]) for row in rows_by_n]
        result[metric] = {
            "values_by_N": values,
            "last_over_first": values[-1] / max(values[0], np.finfo(float).tiny),
            "nonincreasing": bool(values[-1] <= values[0]),
        }
    return result


def _run_pairing(args: argparse.Namespace, pairing_name: str) -> dict[str, Any]:
    contexts = [
        qualification._evaluate_context(
            args,
            pairing_name=pairing_name,
            n=n_value,
            shift=(0.5, 0.5),
        )
        for n_value in args.N_values
    ]
    context_rows: list[dict[str, Any]] = []
    operator_all = True
    ward_all = True
    positive_all = True
    finite_zero_metrics = True
    zero_strict_all = True
    two_plate_positive_all = True
    two_plate_zero_all = True
    for context in contexts:
        cases: dict[str, Any] = {}
        for name, response in context.responses.items():
            states = _response_states(args, response)
            cases[name] = {
                "q_model": response.q_model.tolist(),
                "operator_ward": response.operator_ward.as_dict(),
                "frequencies": states,
            }
            operator_all = operator_all and bool(response.operator_ward.passed)
            ward_all = ward_all and all(row["integrated_ward_passed"] for row in states)
            positive_all = positive_all and all(
                row["pipeline"]["physical_passed"]
                for row in states
                if row["response_sector"] == "positive"
            )
            zero_rows = [row for row in states if row["response_sector"] == "zero"]
            for row in zero_rows:
                strict = row["strict_static"]
                finite_zero_metrics = finite_zero_metrics and bool(
                    strict is not None
                    and all(np.isfinite(float(strict[key])) for key in _STATIC_METRICS)
                )
                zero_strict_all = zero_strict_all and bool(strict and strict["passed"])
        if context.two_plate_batch is None:
            raise RuntimeError("physics smoke context did not retain its two-plate batch")
        plate_2 = qualification._rotated_plate(context.two_plate_batch)
        two_plate = qualification._two_plate_states(
            args,
            context.two_plate_batch.plate_1,
            plate_2,
        )
        for row in two_plate:
            if int(row["n"]) == 0:
                two_plate_zero_all = two_plate_zero_all and bool(row["passed"])
            else:
                two_plate_positive_all = two_plate_positive_all and bool(row["passed"])
        context_rows.append(
            {
                "N": int(context.n),
                "shift": list(context.shift),
                "material_cache": dict(context.material_cache_metadata),
                "execution": dict(context.execution_metadata),
                "cases": cases,
                "two_plate_common_lab": two_plate,
            }
        )

    zero_trends: dict[str, Any] = {}
    for case_name in contexts[0].responses:
        zero_rows = []
        for context_row in context_rows:
            frequencies = context_row["cases"][case_name]["frequencies"]
            zero_rows.append(next(row for row in frequencies if row["response_sector"] == "zero"))
        zero_trends[case_name] = _trend(zero_rows)

    smoke_passed = bool(
        operator_all
        and ward_all
        and positive_all
        and finite_zero_metrics
        and two_plate_positive_all
    )
    return {
        "pairing": pairing_name,
        "contexts": context_rows,
        "zero_mode_strict_static_trends": zero_trends,
        "summary": {
            "operator_identity_all_passed": operator_all,
            "integrated_ward_all_passed": ward_all,
            "positive_pipeline_all_passed": positive_all,
            "zero_mode_metrics_all_finite": finite_zero_metrics,
            "zero_mode_strict_static_all_passed": zero_strict_all,
            "two_plate_positive_all_passed": two_plate_positive_all,
            "two_plate_zero_all_passed": two_plate_zero_all,
            "smoke_passed_without_convergence_claim": smoke_passed,
        },
        "passed": smoke_passed,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _args(argv)
    rows = [_run_pairing(args, pairing) for pairing in args.pairings]
    passed = all(row["passed"] for row in rows)
    payload = {
        "schema": "arbitrary-q-physics-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **source_tree_provenance().as_dict(),
        "config": {
            "pairings": list(args.pairings),
            "N_values": list(args.N_values),
            "workers": int(args.workers),
            "reference_nk": int(args.reference_nk),
            "matsubara_indices": list(args.matsubara_indices),
            "canonical_block_size": int(args.canonical_block_size),
            "runtime_chunk_size": int(args.runtime_chunk_size),
            "temperature_K": float(args.temperature_K),
            "delta0_eV": float(args.delta0_eV),
            "eta_eV": float(args.eta_eV),
            "separation_nm": float(args.separation_nm),
            "ward_tolerance": float(args.ward_tolerance),
            "ward_absolute_tolerance": float(args.ward_absolute_tolerance),
        },
        "interpretation": {
            "convergence_claimed": False,
            "zero_strict_static_required_for_smoke_pass": False,
            "positive_pipeline_and_integrated_ward_required": True,
            "next_stage_if_passed": "real_hardware_formal_performance_preflight",
        },
        "pairings": rows,
        "smoke_passed_without_convergence_claim": bool(passed),
        "formal_numerical_evidence": False,
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": bool(passed),
    }
    _write(args.output, payload)
    print(json.dumps({"output": str(args.output), "passed": passed}, indent=2))
    if not passed:
        raise SystemExit("arbitrary-q physics smoke found an algebraic or positive-pipeline failure")


if __name__ == "__main__":
    main()
