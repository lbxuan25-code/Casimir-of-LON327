"""Timed zero-Matsubara k-grid convergence scan using optimized workspaces.

The scan parallelizes independent ``nk`` values only. Each worker keeps BLAS
threading external to this module so callers can avoid process/thread
oversubscription with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, and related
environment variables.

Besides aggregate static validation fields, the CSV/JSON output resolves the
five longitudinal entries in the scaled local ``(A0,L,T)`` kernel, records the
left/right Ward RHS--collective-projection cancellation, and decomposes the
static ``K_LL`` entry into bubble, direct/contact, bare ``K_SS``, collective
Schur correction, and final effective contributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lno327 import KuboConfig, k_weights, uniform_bz_mesh
from lno327.electrodynamics.static_sheet import static_matsubara_kernel_to_sheet_response
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.ward_validation import validate_effective_ward_xy
from lno327.response.workspace import (
    finite_q_bdg_response_from_q_workspace,
    precompute_finite_q_material_workspace_from_model_ansatz,
    precompute_finite_q_q_workspace,
    primitive_ward_rhs_from_q_workspace,
)
from lno327.workflows.finite_q_engine import FiniteQEngineOptions
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/static_nk_convergence/raw/static_nk_scan.csv"
)
_LONGITUDINAL_COMPONENTS = {
    "k0l": (0, 1),
    "kl0": (1, 0),
    "kll": (1, 1),
    "klt": (1, 2),
    "ktl": (2, 1),
}


def _peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB. This validation CLI targets the project's
    # Linux/WSL environment.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=complex)))


def _scaled_static_kernel(kernel_lt: np.ndarray, energy_scale_eV: float) -> np.ndarray:
    """Mirror the mixed-unit scaling used by the static sheet validation."""

    matrix = np.array(kernel_lt, dtype=complex, copy=True)
    if matrix.shape != (3, 3):
        raise ValueError(f"kernel_lt must have shape (3, 3), got {matrix.shape}")
    energy = float(energy_scale_eV)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy_scale_eV must be finite and positive")
    matrix[0, 0] *= energy
    matrix[1:3, 1:3] /= energy
    return matrix


def _longitudinal_component_diagnostics(
    kernel_lt: np.ndarray,
    energy_scale_eV: float,
) -> dict[str, float | str]:
    """Resolve the aggregate longitudinal norm into its five entries."""

    scaled = _scaled_static_kernel(kernel_lt, energy_scale_eV)
    scale = max(float(np.linalg.norm(scaled.real)), 1.0)
    result: dict[str, float | str] = {"static_kernel_real_scale": scale}
    relative_values: dict[str, float] = {}
    for name, (row, col) in _LONGITUDINAL_COMPONENTS.items():
        absolute = float(abs(scaled[row, col]))
        relative = absolute / scale
        result[f"scaled_abs_{name}"] = absolute
        result[f"relative_{name}"] = relative
        relative_values[name] = relative

    combined = float(np.linalg.norm(np.fromiter(relative_values.values(), dtype=float)))
    dominant = max(relative_values, key=relative_values.__getitem__)
    result["longitudinal_components_relative_norm"] = combined
    result["dominant_longitudinal_component"] = dominant.upper()
    result["dominant_longitudinal_relative"] = relative_values[dominant]
    return result


def _complex_scalar_fields(prefix: str, value: complex, scale: float) -> dict[str, float]:
    scalar = complex(value)
    denominator = max(float(scale), 1e-30)
    return {
        f"{prefix}_real": float(scalar.real),
        f"{prefix}_imag": float(scalar.imag),
        f"{prefix}_abs": float(abs(scalar)),
        f"{prefix}_relative_abs": float(abs(scalar) / denominator),
    }


def _collective_schur_correction(kernel: object) -> np.ndarray:
    """Recompute ``K_Seta K_etaeta^-1 K_etaS`` using the selected inverse policy."""

    k_seta = np.asarray(getattr(kernel, "k_seta"), dtype=complex)
    k_etaeta = np.asarray(getattr(kernel, "k_etaeta"), dtype=complex)
    k_etas = np.asarray(getattr(kernel, "k_etas"), dtype=complex)
    method = str(getattr(kernel, "schur_inverse_method"))
    if method == "inv":
        inverse = np.linalg.inv(k_etaeta)
    elif method == "pinv_diagnostic":
        inverse = np.linalg.pinv(k_etaeta)
    else:
        raise ValueError(f"unsupported Schur inverse method {method!r}")
    return k_seta @ inverse @ k_etas


def _kll_decomposition_diagnostics(
    components: object,
    kernel: object,
    transform: np.ndarray,
    energy_scale_eV: float,
    static_scale: float,
) -> dict[str, float]:
    """Decompose scaled local ``K_LL`` into microscopic and Schur pieces.

    The fixed sign convention is

    ``K_SS = K_bubble + K_direct``

    and

    ``K_eff = K_SS - K_collective_correction``.
    """

    rotation = np.asarray(transform, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"transform must have shape (3, 3), got {rotation.shape}")

    bubble_xy = np.asarray(getattr(components, "bare_bubble"), dtype=complex)
    direct_xy = np.asarray(getattr(components, "direct"), dtype=complex)
    bare_total_xy = np.asarray(getattr(components, "bare_total"), dtype=complex)
    effective_xy = np.asarray(getattr(kernel, "k_eff"), dtype=complex)
    correction_xy = _collective_schur_correction(kernel)

    if not np.allclose(bubble_xy + direct_xy, bare_total_xy, rtol=2e-12, atol=2e-13):
        raise RuntimeError("K_SS decomposition does not satisfy bubble + direct = bare_total")
    if not np.allclose(
        bare_total_xy - correction_xy,
        effective_xy,
        rtol=2e-12,
        atol=2e-13,
    ):
        raise RuntimeError("Schur decomposition does not satisfy K_eff = K_SS - correction")

    pieces_xy = {
        "scaled_kll_bubble": bubble_xy,
        "scaled_kll_direct": direct_xy,
        "scaled_kll_bare_total": bare_total_xy,
        "scaled_kll_collective_correction": correction_xy,
        "scaled_kll_effective": effective_xy,
    }
    values: dict[str, complex] = {}
    result: dict[str, float] = {}
    for prefix, matrix_xy in pieces_xy.items():
        matrix_lt = rotation @ matrix_xy @ rotation.T
        value = complex(_scaled_static_kernel(matrix_lt, energy_scale_eV)[1, 1])
        values[prefix] = value
        result.update(_complex_scalar_fields(prefix, value, static_scale))

    bubble = values["scaled_kll_bubble"]
    direct = values["scaled_kll_direct"]
    bare_total = values["scaled_kll_bare_total"]
    correction = values["scaled_kll_collective_correction"]
    effective = values["scaled_kll_effective"]
    bubble_direct_scale = max(abs(bubble), abs(direct), 1e-30)
    schur_scale = max(abs(bare_total), abs(correction), 1e-30)

    result.update(
        {
            "kll_bubble_direct_cancellation_ratio": float(
                abs(bare_total) / bubble_direct_scale
            ),
            "kll_schur_cancellation_ratio": float(abs(effective) / schur_scale),
            "kll_bubble_direct_closure_abs": float(abs(bubble + direct - bare_total)),
            "kll_schur_closure_abs": float(
                abs(bare_total - correction - effective)
            ),
        }
    )
    return result


def _ward_side_diagnostics(side: object, prefix: str) -> dict[str, float]:
    """Record absolute norms and the physical RHS/projection cancellation ratio."""

    rhs_norm = _norm(getattr(side, "primitive_rhs"))
    projection_norm = _norm(getattr(side, "collective_projection"))
    direct_norm = _norm(getattr(side, "effective_direct"))
    predicted_norm = _norm(getattr(side, "effective_predicted"))
    residual_norm = _norm(getattr(side, "effective_residual"))
    cancellation_scale = max(rhs_norm, projection_norm, 1e-30)
    closure_scale = max(direct_norm, predicted_norm, 1e-30)
    return {
        f"{prefix}_rhs_norm": rhs_norm,
        f"{prefix}_collective_projection_norm": projection_norm,
        f"{prefix}_effective_direct_norm": direct_norm,
        f"{prefix}_effective_predicted_norm": predicted_norm,
        f"{prefix}_effective_residual_norm": residual_norm,
        f"{prefix}_rhs_projection_cancellation_ratio": predicted_norm
        / cancellation_scale,
        f"{prefix}_direct_prediction_relative_residual": residual_norm / closure_scale,
    }


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    nk = int(task["nk"])
    q = np.asarray([task["qx"], task["qy"]], dtype=float)
    cpu_start = time.process_time()
    total_start = time.perf_counter()

    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(task["pairing"], phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(task["delta0_eV"])
    points = uniform_bz_mesh(nk)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.0,
        temperature_K=task["temperature_K"],
        eta_eV=task["eta_eV"],
        output_si=False,
    )
    options = FiniteQEngineOptions()

    start = time.perf_counter()
    material = precompute_finite_q_material_workspace_from_model_ansatz(
        model.spec,
        ansatz,
        points,
        weights,
        config,
        pairing,
        options,
    )
    material_seconds = time.perf_counter() - start

    start = time.perf_counter()
    q_workspace = precompute_finite_q_q_workspace(material, q)
    q_workspace_seconds = time.perf_counter() - start

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        components = finite_q_bdg_response_from_q_workspace(q_workspace, 0.0)
    response_seconds = time.perf_counter() - start

    start = time.perf_counter()
    kernel = effective_em_kernel_from_components(components, q_model=q, xi_eV=0.0)
    rhs = primitive_ward_rhs_from_q_workspace(q_workspace, 0.0)
    ward = validate_effective_ward_xy(
        kernel,
        rhs,
        residual_tolerance=task["ward_tolerance"],
    )
    static = static_matsubara_kernel_to_sheet_response(kernel, ward)
    postprocess_seconds = time.perf_counter() - start

    longitudinal = _longitudinal_component_diagnostics(
        static.kernel_lt,
        static.energy_scale_eV,
    )
    transform = np.asarray(static.metadata["local_projection_matrix"], dtype=float)
    kll_decomposition = _kll_decomposition_diagnostics(
        components,
        kernel,
        transform,
        static.energy_scale_eV,
        float(longitudinal["static_kernel_real_scale"]),
    )
    ward_left = _ward_side_diagnostics(ward.left, "ward_left")
    ward_right = _ward_side_diagnostics(ward.right, "ward_right")

    # Keep all decompositions tied to the production aggregate definition.
    if not np.isclose(
        longitudinal["longitudinal_components_relative_norm"],
        static.validation.relative_longitudinal_gauge_residual,
        rtol=5e-13,
        atol=5e-15,
    ):
        raise RuntimeError("longitudinal component decomposition disagrees with static validation")
    if not np.isclose(
        kll_decomposition["scaled_kll_effective_relative_abs"],
        longitudinal["relative_kll"],
        rtol=5e-13,
        atol=5e-15,
    ):
        raise RuntimeError("K_LL decomposition disagrees with longitudinal diagnostics")

    total_seconds = time.perf_counter() - total_start
    cpu_seconds = time.process_time() - cpu_start
    warning_messages = [str(item.message) for item in caught]

    return {
        "nk": nk,
        "num_k_points": int(points.shape[0]),
        "pairing": task["pairing"],
        "qx": float(q[0]),
        "qy": float(q[1]),
        "temperature_K": float(task["temperature_K"]),
        "delta0_eV": float(task["delta0_eV"]),
        "eta_eV": float(task["eta_eV"]),
        "ward_tolerance": float(task["ward_tolerance"]),
        "material_seconds": material_seconds,
        "q_workspace_seconds": q_workspace_seconds,
        "response_seconds": response_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_wall_seconds": total_seconds,
        "process_cpu_seconds": cpu_seconds,
        "peak_rss_mb": _peak_rss_mb(),
        "midpoint_eigensystems": int(material.metadata["midpoint_eigensystem_count"]),
        "shifted_eigensystems": int(q_workspace.metadata["shifted_eigensystem_count"]),
        "ward_left_primitive": ward.left.primitive_relative_residual,
        "ward_right_primitive": ward.right.primitive_relative_residual,
        "ward_left_effective": ward.left.effective_relative_residual,
        "ward_right_effective": ward.right.effective_relative_residual,
        "schur_condition_number": ward.schur_condition_number,
        "ward_passed": bool(ward.passed),
        **ward_left,
        **ward_right,
        "relative_imaginary_norm": static.validation.relative_imaginary_norm,
        "relative_longitudinal_gauge_residual": (
            static.validation.relative_longitudinal_gauge_residual
        ),
        **longitudinal,
        **kll_decomposition,
        "relative_density_transverse_mixing": (
            static.validation.relative_density_transverse_mixing
        ),
        "chi_bar": static.chi_bar,
        "dbar_t": static.dbar_t,
        "static_validation_passed": bool(static.validation.passed),
        "warning_count": len(warning_messages),
        "warning_first": warning_messages[0] if warning_messages else "",
        "pid": os.getpid(),
    }


def _write_outputs(rows: list[dict[str, Any]], output: Path, args: argparse.Namespace) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = output.with_suffix(".json")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "workers": args.workers,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "diagnostic_definitions": {
            "scaled_abs_k0l": "|scaled K_0L|",
            "scaled_abs_kl0": "|scaled K_L0|",
            "scaled_abs_kll": "|scaled K_LL|",
            "scaled_abs_klt": "|scaled K_LT|",
            "scaled_abs_ktl": "|scaled K_TL|",
            "relative_kxy": "scaled_abs_kxy / max(||Re K_scaled||_F, 1)",
            "kll_sign_convention": (
                "K_SS = K_bubble + K_direct; "
                "K_eff = K_SS - K_collective_correction"
            ),
            "kll_bubble_direct_cancellation_ratio": (
                "|K_LL^SS| / max(|K_LL^bubble|, |K_LL^direct|, 1e-30)"
            ),
            "kll_schur_cancellation_ratio": (
                "|K_LL^eff| / max(|K_LL^SS|, |K_LL^collective|, 1e-30)"
            ),
            "rhs_projection_cancellation_ratio": (
                "||effective_predicted|| / max(||primitive_rhs||, "
                "||collective_projection||, 1e-30)"
            ),
        },
        "rows": rows,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    header = (
        " nk    Nk      total[s]  q-cache[s]  Ward-eff(max)  longitudinal  "
        "L-dominant   rel(L-dom)    chi_bar      Dbar_T"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ward_eff = max(row["ward_left_effective"], row["ward_right_effective"])
        print(
            f"{row['nk']:3d} {row['num_k_points']:6d} "
            f"{row['total_wall_seconds']:11.4f} "
            f"{row['q_workspace_seconds']:10.4f} "
            f"{ward_eff:13.3e} "
            f"{row['relative_longitudinal_gauge_residual']:12.3e} "
            f"{row['dominant_longitudinal_component']:>10s} "
            f"{row['dominant_longitudinal_relative']:12.3e} "
            f"{row['chi_bar']:12.5e} "
            f"{row['dbar_t']:12.5e}"
        )

    print("\nLongitudinal decomposition (each entry uses the aggregate static scale)")
    header = " nk      rel(K0L)     rel(KL0)     rel(KLL)     rel(KLT)     rel(KTL)"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['nk']:3d} "
            f"{row['relative_k0l']:13.3e} "
            f"{row['relative_kl0']:13.3e} "
            f"{row['relative_kll']:13.3e} "
            f"{row['relative_klt']:13.3e} "
            f"{row['relative_ktl']:13.3e}"
        )

    print("\nK_LL decomposition (scaled local values; K_eff = K_SS - K_coll)")
    header = (
        " nk       Re(bubble)    Re(direct)       Re(KSS)      Re(Kcoll)      "
        "Re(Keff)  b+d cancel  Schur cancel"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['nk']:3d} "
            f"{row['scaled_kll_bubble_real']:14.6e} "
            f"{row['scaled_kll_direct_real']:14.6e} "
            f"{row['scaled_kll_bare_total_real']:14.6e} "
            f"{row['scaled_kll_collective_correction_real']:14.6e} "
            f"{row['scaled_kll_effective_real']:14.6e} "
            f"{row['kll_bubble_direct_cancellation_ratio']:11.3e} "
            f"{row['kll_schur_cancellation_ratio']:13.3e}"
        )

    print("\nWard RHS--collective cancellation")
    header = (
        " nk     L:||R||      L:||P||   L:cancel     "
        "R:||R||      R:||P||   R:cancel"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['nk']:3d} "
            f"{row['ward_left_rhs_norm']:12.3e} "
            f"{row['ward_left_collective_projection_norm']:12.3e} "
            f"{row['ward_left_rhs_projection_cancellation_ratio']:10.3e} "
            f"{row['ward_right_rhs_norm']:12.3e} "
            f"{row['ward_right_collective_projection_norm']:12.3e} "
            f"{row['ward_right_rhs_projection_cancellation_ratio']:10.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nks", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pairing", choices=("spm", "dwave"), default="spm")
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    if any(nk <= 0 for nk in args.nks):
        parser.error("all --nks values must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not np.isfinite([args.qx, args.qy]).all() or np.hypot(args.qx, args.qy) == 0.0:
        parser.error("(qx, qy) must be finite and nonzero")

    common = {
        "pairing": args.pairing,
        "qx": args.qx,
        "qy": args.qy,
        "temperature_K": args.temperature_K,
        "delta0_eV": args.delta0_eV,
        "eta_eV": args.eta_eV,
        "ward_tolerance": args.ward_tolerance,
    }
    tasks = [{**common, "nk": nk} for nk in sorted(set(args.nks))]

    sweep_start = time.perf_counter()
    if args.workers == 1:
        rows = [_run_one(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = {executor.submit(_run_one, task): task["nk"] for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"completed nk={row['nk']} in {row['total_wall_seconds']:.3f} s "
                    f"(peak RSS {row['peak_rss_mb']:.1f} MiB)",
                    flush=True,
                )
        rows.sort(key=lambda row: row["nk"])
    sweep_seconds = time.perf_counter() - sweep_start

    _write_outputs(rows, args.output, args)
    _print_summary(rows)
    print(f"\nSweep wall time: {sweep_seconds:.4f} s")
    print(f"CSV:  {args.output}")
    print(f"JSON: {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
