"""Test actual-shift FS masks and signed BdG-pair reconstruction for d-wave response."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from validation.lib.dwave_shift_batch import ShiftBatchConfig, postprocess_merged
from validation.lib.dwave_shift_signed_reconstruction import (
    aggregate_rule_signed_summaries,
    signed_pair_reconstruction_residuals,
    signed_reconstruction_residuals,
    summarize_shift_signed_reconstruction,
)
from validation.lib.dwave_shift_spatial import (
    SpatialDiagnosticConfig,
    components_from_primitive_vector,
    evaluate_shift_spatial,
    shift_rule,
)


DEFAULT_OUTPUT = Path(
    "validation/outputs/zero_matsubara/dwave_shift_signed/raw/"
    "dwave_q003_002_base56_gauss2_vs_halton4.json"
)


def _shift_key(shift: Sequence[float]) -> tuple[float, float]:
    value = np.asarray(shift, dtype=float)
    return round(float(value[0]), 15), round(float(value[1]), 15)


def _evaluate_portable(
    config: SpatialDiagnosticConfig,
    shift: np.ndarray,
    shell_multiples_T: tuple[float, ...],
) -> dict[str, Any]:
    result = evaluate_shift_spatial(config, shift, keep_workspace=True)
    summary = summarize_shift_signed_reconstruction(
        result["workspace"],
        result["vectors"],
        shell_multiples_T=shell_multiples_T,
    )
    return {"shift": np.asarray(result["shift"], dtype=float), "summary": summary}


def _physical_config(args) -> ShiftBatchConfig:
    return ShiftBatchConfig(
        base_nk=args.base_nk,
        qx=args.qx,
        qy=args.qy,
        temperature_K=args.temperature_K,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta_eV,
        ward_tolerance=args.ward_tolerance,
        ward_absolute_tolerance=args.ward_absolute_tolerance,
        condition_max=args.condition_max,
        raw_longitudinal_ceiling=args.raw_longitudinal_ceiling,
        longitudinal_tolerance=args.longitudinal_tolerance,
        mixing_tolerance=args.mixing_tolerance,
        reality_tolerance=args.reality_tolerance,
        passivity_tolerance=args.passivity_tolerance,
        separation_nm=args.separation_nm,
    )


def _postprocess_vector(vector, workspace, physical_config):
    components, rhs = components_from_primitive_vector(vector, workspace)
    return postprocess_merged(components, rhs, physical_config)


def _relative(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-30)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _spatial_rows(
    total_a: np.ndarray,
    total_b: np.ndarray,
    rule_a: Mapping[str, Any],
    rule_b: Mapping[str, Any],
    workspace,
    physical,
    result_b: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target_delta = total_b - total_a
    rows: list[dict[str, Any]] = []
    names = sorted(set(rule_a["spatial_sums"]) & set(rule_b["spatial_sums"]))
    for name in names:
        selected_delta = (
            np.asarray(rule_b["spatial_sums"][name], dtype=complex)
            - np.asarray(rule_a["spatial_sums"][name], dtype=complex)
        )
        residuals = signed_reconstruction_residuals(target_delta, selected_delta)
        reconstructed = _postprocess_vector(total_a + selected_delta, workspace, physical)
        rows.append(
            {
                "class": name,
                "rule_a_point_fraction": float(rule_a["spatial_point_fractions"][name]),
                "rule_b_point_fraction": float(rule_b["spatial_point_fractions"][name]),
                "k_ss_relative_residual": residuals["k_ss"],
                "k_seta_relative_residual": residuals["k_seta"],
                "k_etas_relative_residual": residuals["k_etas"],
                "k_etaeta_relative_residual": residuals["k_etaeta"],
                "ward_rhs_relative_residual": residuals["ward_rhs"],
                "chi_bar": reconstructed["chi_bar"],
                "dbar_t": reconstructed["dbar_t"],
                "relative_chi_residual_to_rule_b": _relative(
                    reconstructed["chi_bar"], result_b["chi_bar"]
                ),
                "relative_dbar_residual_to_rule_b": _relative(
                    reconstructed["dbar_t"], result_b["dbar_t"]
                ),
                "ward_primitive_mixed_ratio_max": reconstructed[
                    "ward_primitive_mixed_ratio_max"
                ],
                "ward_effective_mixed_ratio_max": reconstructed[
                    "ward_effective_mixed_ratio_max"
                ],
                "raw_longitudinal": reconstructed["raw_longitudinal"],
                "projection_eligible": reconstructed["projection_eligible"],
            }
        )
    return rows


def _pair_rows(
    total_a: np.ndarray,
    rule_a: Mapping[str, Any],
    rule_b: Mapping[str, Any],
    workspace,
    physical,
    result_b: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target_pair_delta = (
        np.asarray(rule_b["pair_sums"]["all_pairs"], dtype=complex)
        - np.asarray(rule_a["pair_sums"]["all_pairs"], dtype=complex)
    )
    rows: list[dict[str, Any]] = []
    names = sorted(set(rule_a["pair_sums"]) & set(rule_b["pair_sums"]))
    for name in names:
        selected_delta = (
            np.asarray(rule_b["pair_sums"][name], dtype=complex)
            - np.asarray(rule_a["pair_sums"][name], dtype=complex)
        )
        residuals = signed_pair_reconstruction_residuals(target_pair_delta, selected_delta)
        reconstructed = _postprocess_vector(total_a + selected_delta, workspace, physical)
        rows.append(
            {
                "class": name,
                "rule_a_pair_event_fraction": float(rule_a["pair_event_fractions"][name]),
                "rule_b_pair_event_fraction": float(rule_b["pair_event_fractions"][name]),
                **{f"{key}_relative_residual": value for key, value in residuals.items()},
                "chi_bar": reconstructed["chi_bar"],
                "dbar_t": reconstructed["dbar_t"],
                "relative_chi_residual_to_rule_b": _relative(
                    reconstructed["chi_bar"], result_b["chi_bar"]
                ),
                "relative_dbar_residual_to_rule_b": _relative(
                    reconstructed["dbar_t"], result_b["dbar_t"]
                ),
                "ward_primitive_mixed_ratio_max": reconstructed[
                    "ward_primitive_mixed_ratio_max"
                ],
                "raw_longitudinal": reconstructed["raw_longitudinal"],
                "bubble_only_reconstruction": True,
            }
        )
    return rows


def _best_spatial_candidate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    candidates = [row for row in rows if row["class"] != "all_points"]
    if not candidates:
        return None, float("inf")

    def score(row: Mapping[str, Any]) -> float:
        return max(
            float(row["k_ss_relative_residual"]),
            float(row["k_seta_relative_residual"]),
            float(row["k_etas_relative_residual"]),
            float(row["k_etaeta_relative_residual"]),
            float(row["ward_rhs_relative_residual"]),
            float(row["relative_chi_residual_to_rule_b"]),
            float(row["relative_dbar_residual_to_rule_b"]),
        )

    best = min(candidates, key=score)
    return best, score(best)


def _best_pair_candidate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    candidates = [row for row in rows if row["class"] != "all_pairs"]
    if not candidates:
        return None, float("inf")

    def score(row: Mapping[str, Any]) -> float:
        return max(
            float(row["k_ss_bubble_relative_residual"]),
            float(row["k_seta_bubble_relative_residual"]),
            float(row["k_etas_bubble_relative_residual"]),
            float(row["k_etaeta_bubble_relative_residual"]),
        )

    best = min(candidates, key=score)
    return best, score(best)


def _summary_text(
    args,
    result_a: Mapping[str, Any],
    result_b: Mapping[str, Any],
    spatial_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    wall_seconds: float,
) -> str:
    lines = [
        "d-wave actual-shift signed reconstruction diagnostic",
        "=" * 59,
        f"q = ({args.qx:.8g}, {args.qy:.8g}), base_nk = {args.base_nk}",
        f"rule A = {args.rule_a}; rule B = {args.rule_b}",
        f"T = {args.temperature_K:.8g} K; wall time = {wall_seconds:.3f} s",
        "",
        "Rule totals",
        "-----------",
        (
            f"A: chi_bar={result_a['chi_bar']:.10g}, Dbar_T={result_a['dbar_t']:.10g}, "
            f"Ward_prim={result_a['ward_primitive_mixed_ratio_max']:.3e}, "
            f"raw_long={result_a['raw_longitudinal']:.3e}"
        ),
        (
            f"B: chi_bar={result_b['chi_bar']:.10g}, Dbar_T={result_b['dbar_t']:.10g}, "
            f"Ward_prim={result_b['ward_primitive_mixed_ratio_max']:.3e}, "
            f"raw_long={result_b['raw_longitudinal']:.3e}"
        ),
        "",
        "Actual-shift full-primitive spatial reconstruction",
        "--------------------------------------------------",
        (
            "class                           fracA  fracB  rSS   rSeta rEtaS rEtaEta "
            "rRHS  rChi  rDT   Ward"
        ),
    ]
    for row in spatial_rows:
        lines.append(
            f"{row['class']:<31s} "
            f"{float(row['rule_a_point_fraction']):6.3f} "
            f"{float(row['rule_b_point_fraction']):6.3f} "
            f"{float(row['k_ss_relative_residual']):5.3f} "
            f"{float(row['k_seta_relative_residual']):6.3f} "
            f"{float(row['k_etas_relative_residual']):5.3f} "
            f"{float(row['k_etaeta_relative_residual']):7.3f} "
            f"{float(row['ward_rhs_relative_residual']):5.3f} "
            f"{float(row['relative_chi_residual_to_rule_b']):5.3f} "
            f"{float(row['relative_dbar_residual_to_rule_b']):5.3f} "
            f"{float(row['ward_primitive_mixed_ratio_max']):.3e}"
        )
    lines.extend(
        [
            "",
            "Signed BdG-pair bubble reconstruction",
            "--------------------------------------",
            "class                              fracA    fracB   rSS   rSeta rEtaS rEtaEta rChi rDT Ward",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"{row['class']:<34s} "
            f"{float(row['rule_a_pair_event_fraction']):7.4f} "
            f"{float(row['rule_b_pair_event_fraction']):7.4f} "
            f"{float(row['k_ss_bubble_relative_residual']):5.3f} "
            f"{float(row['k_seta_bubble_relative_residual']):6.3f} "
            f"{float(row['k_etas_bubble_relative_residual']):5.3f} "
            f"{float(row['k_etaeta_bubble_relative_residual']):7.3f} "
            f"{float(row['relative_chi_residual_to_rule_b']):5.3f} "
            f"{float(row['relative_dbar_residual_to_rule_b']):5.3f} "
            f"{float(row['ward_primitive_mixed_ratio_max']):.3e}"
        )

    best_spatial, spatial_score = _best_spatial_candidate(spatial_rows)
    best_pair, pair_score = _best_pair_candidate(pair_rows)
    lines.extend(["", "Corrected solver-direction verdict", "---------------------------------"])
    if best_spatial is None:
        lines.append("No nontrivial actual-shift spatial class was available.")
    elif spatial_score <= 0.20 and float(best_spatial["ward_primitive_mixed_ratio_max"]) < 1.0:
        lines.append(
            f"The full-primitive class '{best_spatial['class']}' reconstructs all monitored blocks "
            f"and physical channels to within {spatial_score:.3f} relative residual while preserving Ward. "
            "A symmetry-complete strip subtraction prototype is justified."
        )
    elif spatial_score <= 0.40:
        lines.append(
            f"The best full-primitive class is '{best_spatial['class']}' with worst residual "
            f"{spatial_score:.3f}. Shifted-FS structure is relevant but not yet a compact complete correction."
        )
    else:
        lines.append(
            f"The best full-primitive class is '{best_spatial['class']}', but its worst residual is "
            f"{spatial_score:.3f}. No tested compact shifted-FS mask reliably reconstructs the rule difference."
        )
    if best_pair is not None:
        lines.append(
            f"Best signed pair class: '{best_pair['class']}' with worst bubble-block residual "
            f"{pair_score:.3f}. This is bubble-only and is not by itself a Ward-closed correction."
        )
    lines.extend(
        [
            "",
            "Audits:",
            "- all_points must reconstruct the complete rule-B primitive response from rule A.",
            "- all_pairs must reconstruct the complete signed bubble difference.",
            "- pair rows omit direct/contact, counterterm, phase-direct, and full Ward-RHS corrections.",
            "",
            "This diagnostic does not itself grant Casimir eligibility.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-nk", type=int, default=56)
    parser.add_argument("--rule-a", choices=["midpoint", "gauss2", "halton4"], default="gauss2")
    parser.add_argument("--rule-b", choices=["midpoint", "gauss2", "halton4"], default="halton4")
    parser.add_argument("--fs-shell-T", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--qx", type=float, default=0.03)
    parser.add_argument("--qy", type=float, default=0.02)
    parser.add_argument("--temperature-K", type=float, default=10.0)
    parser.add_argument("--delta0-eV", type=float, default=0.1)
    parser.add_argument("--eta-eV", type=float, default=1e-8)
    parser.add_argument("--ward-tolerance", type=float, default=1e-7)
    parser.add_argument("--ward-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--condition-max", type=float, default=1e12)
    parser.add_argument("--raw-longitudinal-ceiling", type=float, default=1e-3)
    parser.add_argument("--longitudinal-tolerance", type=float, default=1e-7)
    parser.add_argument("--mixing-tolerance", type=float, default=1e-7)
    parser.add_argument("--reality-tolerance", type=float, default=1e-9)
    parser.add_argument("--passivity-tolerance", type=float, default=1e-10)
    parser.add_argument("--separation-nm", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.base_nk <= 0 or args.workers <= 0:
        raise ValueError("base-nk and workers must be positive")
    if args.rule_a == args.rule_b:
        raise ValueError("rule-a and rule-b must differ")
    if any(not np.isfinite(value) or value <= 0.0 for value in args.fs_shell_T):
        raise ValueError("fs-shell-T values must be positive and finite")

    started = time.perf_counter()
    config = SpatialDiagnosticConfig(
        base_nk=args.base_nk,
        qx=args.qx,
        qy=args.qy,
        temperature_K=args.temperature_K,
        delta0_eV=args.delta0_eV,
        eta_eV=args.eta_eV,
    )
    shifts_a, weights_a = shift_rule(args.rule_a)
    shifts_b, weights_b = shift_rule(args.rule_b)
    all_shifts: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for shift in np.concatenate([shifts_a, shifts_b], axis=0):
        key = _shift_key(shift)
        if key not in seen:
            all_shifts.append(np.asarray(shift, dtype=float))
            seen.add(key)

    first_raw = evaluate_shift_spatial(config, all_shifts[0], keep_workspace=True)
    workspace = first_raw["workspace"]
    first_summary = summarize_shift_signed_reconstruction(
        workspace,
        first_raw["vectors"],
        shell_multiples_T=tuple(args.fs_shell_T),
    )
    cache = {_shift_key(first_raw["shift"]): first_summary}
    remaining = all_shifts[1:]
    if args.workers == 1:
        for index, shift in enumerate(remaining, start=2):
            result = _evaluate_portable(config, shift, tuple(args.fs_shell_T))
            cache[_shift_key(result["shift"])] = result["summary"]
            print(f"completed grid {index}/{len(all_shifts)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_evaluate_portable, config, shift, tuple(args.fs_shell_T)): shift
                for shift in remaining
            }
            completed = 1
            for future in as_completed(futures):
                result = future.result()
                cache[_shift_key(result["shift"])] = result["summary"]
                completed += 1
                print(f"completed grid {completed}/{len(all_shifts)}")

    rule_a = aggregate_rule_signed_summaries(
        shifts_a, weights_a, cache, key_function=_shift_key
    )
    rule_b = aggregate_rule_signed_summaries(
        shifts_b, weights_b, cache, key_function=_shift_key
    )
    total_a = np.asarray(rule_a["spatial_sums"]["all_points"], dtype=complex)
    total_b = np.asarray(rule_b["spatial_sums"]["all_points"], dtype=complex)
    physical = _physical_config(args)
    result_a = _postprocess_vector(total_a, workspace, physical)
    result_b = _postprocess_vector(total_b, workspace, physical)
    spatial_rows = _spatial_rows(
        total_a, total_b, rule_a, rule_b, workspace, physical, result_b
    )
    pair_rows = _pair_rows(total_a, rule_a, rule_b, workspace, physical, result_b)

    all_points = next(row for row in spatial_rows if row["class"] == "all_points")
    if max(
        float(all_points["k_ss_relative_residual"]),
        float(all_points["k_seta_relative_residual"]),
        float(all_points["k_etas_relative_residual"]),
        float(all_points["k_etaeta_relative_residual"]),
        float(all_points["ward_rhs_relative_residual"]),
    ) > 1e-10:
        raise RuntimeError("all_points signed primitive reconstruction audit failed")
    all_pairs = next(row for row in pair_rows if row["class"] == "all_pairs")
    if max(
        float(all_pairs["k_ss_bubble_relative_residual"]),
        float(all_pairs["k_seta_bubble_relative_residual"]),
        float(all_pairs["k_etas_bubble_relative_residual"]),
        float(all_pairs["k_etaeta_bubble_relative_residual"]),
    ) > 1e-10:
        raise RuntimeError("all_pairs signed bubble reconstruction audit failed")

    wall_seconds = time.perf_counter() - started
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    summary_path = output.with_name(stem + ".summary.txt")
    spatial_path = output.with_name(stem + ".spatial.csv")
    pair_path = output.with_name(stem + ".pairs.csv")
    _write_csv(spatial_path, spatial_rows)
    _write_csv(pair_path, pair_rows)
    summary = _summary_text(
        args, result_a, result_b, spatial_rows, pair_rows, wall_seconds
    )
    summary_path.write_text(summary, encoding="utf-8")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "contract": (
            "actual-shift normal-state FS masks; signed complex BdG-pair bubble decomposition; "
            "full primitive masked reconstruction before one Schur; diagnostic only"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "rule_a": {"shifts": shifts_a.tolist(), "weights": weights_a.tolist(), "result": result_a},
        "rule_b": {"shifts": shifts_b.tolist(), "weights": weights_b.tolist(), "result": result_b},
        "spatial_reconstruction": spatial_rows,
        "pair_reconstruction": pair_rows,
        "wall_seconds": wall_seconds,
        "files": {
            "summary_txt": str(summary_path),
            "spatial_csv": str(spatial_path),
            "pairs_csv": str(pair_path),
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSigned reconstruction diagnostic completed.")
    print(
        f"Rule A: chi={result_a['chi_bar']:.7f}, D_T={result_a['dbar_t']:.7f}, "
        f"Ward={result_a['ward_primitive_mixed_ratio_max']:.3e}"
    )
    print(
        f"Rule B: chi={result_b['chi_bar']:.7f}, D_T={result_b['dbar_t']:.7f}, "
        f"Ward={result_b['ward_primitive_mixed_ratio_max']:.3e}"
    )
    print(f"Summary: {summary_path}")
    print(f"Spatial: {spatial_path}")
    print(f"Pairs:   {pair_path}")
    print(f"JSON:    {output}")


if __name__ == "__main__":
    main()
