#!/usr/bin/env python3
"""Compare diagnostic BdG finite-q quadrature strategies at kernel level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bdg_quadrature_strategy_common import (
    PAIRINGS,
    composite_uniform_quadrature,
    compute_bdg_components_for_composite_grid,
    recommend_strategy,
    strategy_case_status,
    strategy_origins,
)
from lno327.conductivity import KuboConfig


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "validation" / "outputs" / "response" / "bdg_finite_q"
Q_CASES = ((0.01, 0.0), (0.01, 0.01))
N_LIST_QUICK = (24, 36, 48)
N_LIST_FULL = (24, 36, 48, 72, 96)
HIGH_RESOLUTION_N = (48, 72, 96)
MULTI_ORIGIN_DENSE_N = (36, 48, 72)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _strategy_specs(quick: bool) -> list[tuple[str, int, tuple[float, float]]]:
    base_n = N_LIST_QUICK if quick else N_LIST_FULL
    specs: list[tuple[str, int, tuple[float, float]]] = []
    for strategy in ("ordinary_uniform", "multi_origin_symmetric"):
        specs.extend((strategy, n_grid, q) for n_grid in base_n for q in Q_CASES)
    for n_grid in base_n:
        q_component = 4.0 * np.pi / float(n_grid)
        specs.extend(
            [
                ("grid_step_commensurate_reference", n_grid, (q_component, 0.0)),
                ("grid_step_commensurate_reference", n_grid, (q_component, q_component)),
            ]
        )
    specs.extend(
        ("high_resolution_uniform", n_grid, q)
        for n_grid in HIGH_RESOLUTION_N
        for q in Q_CASES
    )
    specs.extend(
        ("multi_origin_dense", n_grid, q)
        for n_grid in MULTI_ORIGIN_DENSE_N
        for q in Q_CASES
    )
    return specs


def _case_row(
    pairing: str,
    strategy: str,
    n_grid: int,
    q_model: tuple[float, float],
    origins: list[tuple[float, float]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "pairing": pairing,
        "strategy": strategy,
        "N": int(n_grid),
        "num_origins": len(origins),
        "num_k_points_total": int(n_grid * n_grid * len(origins)),
        "q_model": list(q_model),
        "omega_eV": 0.01,
        "Vx": metrics["contact_closure"]["Vx"],
        "Vy": metrics["contact_closure"]["Vy"],
        "bare_total_ward_max_abs": metrics["bare_total_ward_max_abs"],
        "amplitude_phase_ward_max_abs": metrics["amplitude_phase_ward_max_abs"],
        "collective_condition_number": metrics["collective_condition_number"],
        "collective_inverse_method": metrics["collective_inverse_method"],
        "sigma_diag_min_real": metrics["sigma_diag_min_real"],
        "sigma_offdiag_rel": metrics["sigma_offdiag_rel"],
        "sigma_xx_yy_anisotropy": metrics["sigma_xx_yy_anisotropy"],
        "max_abs_sigma_tilde": metrics["max_abs_sigma_tilde"],
    }
    row["contact_closure_max_abs"] = max(
        float(row[channel]["E_band_plus_qD_abs"]) for channel in ("Vx", "Vy")
    )
    row["status"], row["dominant_failure"] = strategy_case_status(row)
    return row


def _best_observed_strategy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {"ordinary_uniform", "multi_origin_symmetric", "high_resolution_uniform", "multi_origin_dense"}
    candidates: list[dict[str, Any]] = []
    for strategy, n_grid in sorted(
        {(row["strategy"], int(row["N"])) for row in rows if row["strategy"] in allowed}
    ):
        onsite = [
            row
            for row in rows
            if row["pairing"] == "onsite_s" and row["strategy"] == strategy and int(row["N"]) == n_grid
        ]
        material = [
            row
            for row in rows
            if row["pairing"] in {"spm", "dwave"}
            and row["strategy"] == strategy
            and int(row["N"]) == n_grid
        ]
        candidates.append(
            {
                "strategy": strategy,
                "N": n_grid,
                "num_origins": int(onsite[0]["num_origins"]),
                "num_k_points_total": int(onsite[0]["num_k_points_total"]),
                "onsite_s_amplitude_phase_ward_max_abs": max(
                    float(row["amplitude_phase_ward_max_abs"]) for row in onsite
                ),
                "onsite_s_contact_closure_max_abs": max(
                    float(row[channel]["E_band_plus_qD_abs"])
                    for row in onsite
                    for channel in ("Vx", "Vy")
                ),
                "onsite_s_bare_ward_max_abs": max(
                    float(row["bare_total_ward_max_abs"]) for row in onsite
                ),
                "material_amplitude_phase_ward_max_abs": max(
                    float(row["amplitude_phase_ward_max_abs"]) for row in material
                ),
            }
        )
    return min(
        candidates,
        key=lambda row: (
            row["onsite_s_amplitude_phase_ward_max_abs"],
            row["onsite_s_contact_closure_max_abs"],
            row["material_amplitude_phase_ward_max_abs"],
            row["num_k_points_total"],
        ),
    )


def build_payload(quick: bool = True) -> dict[str, Any]:
    cfg = KuboConfig.from_kelvin(
        omega_eV=0.01,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for pairing in PAIRINGS:
        for strategy, n_grid, q_model in _strategy_specs(quick):
            q = np.asarray(q_model, dtype=float)
            origins = strategy_origins(strategy, q)
            origin_key = tuple((float(x), float(y)) for x, y in origins)
            key = (pairing, int(n_grid), tuple(float(value) for value in q), origin_key)
            if key not in cache:
                points, weights = composite_uniform_quadrature(n_grid, origins)
                cache[key] = compute_bdg_components_for_composite_grid(
                    pairing,
                    0.01,
                    q,
                    points,
                    weights,
                    cfg,
                    delta0_eV=0.04,
                )
            rows.append(_case_row(pairing, strategy, n_grid, q_model, origins, cache[key]))

    recommended = recommend_strategy(rows)
    best_observed = _best_observed_strategy(rows)
    if recommended is None:
        status = "FAILED"
        dominant_failure = "bare_ward"
        best_interpretation = (
            "No tested arbitrary-q strategy simultaneously reaches onsite_s contact, bare-Ward, and AP-Ward 1e-6 limits."
        )
        no_recommendation_reason = best_interpretation
    else:
        status = "PASSED"
        dominant_failure = "none"
        best_interpretation = (
            f"{recommended['strategy']} at N={recommended['N']} is the best passing validation-mode candidate."
        )
        no_recommendation_reason = None
    summary = {
        "status": status,
        "formal_casimir_ran": False,
        "diagnostic_only": True,
        "dominant_failure": dominant_failure,
        "best_interpretation": best_interpretation,
        "best_observed_strategy": best_observed,
        "recommended_strategy_for_stageSC_2b": recommended,
        "no_recommendation_reason": no_recommendation_reason,
    }
    return {
        **summary,
        "quick": bool(quick),
        "schur_assembly": "one Schur inversion after composite linear-kernel accumulation",
        "production_default_modified": False,
        "summary": summary,
        "cases": rows,
    }


def _q_label(row: dict[str, Any]) -> str:
    return f"({row['q_model'][0]:.7g},{row['q_model'][1]:.7g})"


def _write_report(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "stageSC_0f_bdg_quadrature_strategy_comparison_audit"
    json_path = OUTPUT_DIR / f"{stem}.json"
    md_path = OUTPUT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {stem}",
        "",
        f"- status: {payload['status']}",
        f"- formal Casimir ran: {payload['formal_casimir_ran']}",
        f"- diagnostic only: {payload['diagnostic_only']}",
        f"- Schur assembly: {payload['schur_assembly']}",
        f"- interpretation: {payload['best_interpretation']}",
        (
            f"- best observed (not necessarily passing): {payload['best_observed_strategy']['strategy']} "
            f"N={payload['best_observed_strategy']['N']}"
        ),
        "",
        "## Strategy comparison summary",
        "",
        "| pairing | strategy | N | q | contact closure max | bare Ward max | AP Ward max | status |",
        "| ------- | -------- | -: | - | ------------------: | ------------: | ----------: | ------ |",
    ]
    for row in payload["cases"]:
        lines.append(
            f"| {row['pairing']} | {row['strategy']} | {row['N']} | {_q_label(row)} | "
            f"{row['contact_closure_max_abs']:.8g} | {row['bare_total_ward_max_abs']:.8g} | "
            f"{row['amplitude_phase_ward_max_abs']:.8g} | {row['status']} |"
        )

    onsite = [
        row
        for row in payload["cases"]
        if row["pairing"] == "onsite_s" and row["strategy"] != "grid_step_commensurate_reference"
    ]
    onsite.sort(
        key=lambda row: (
            float(row["amplitude_phase_ward_max_abs"]),
            float(row["contact_closure_max_abs"]),
            int(row["num_k_points_total"]),
        )
    )
    lines.extend(
        [
            "",
            "## onsite_s fixed-q strategy ranking",
            "",
            "| rank | strategy | N | origins | contact closure | bare Ward | AP Ward | cost |",
            "| ---: | -------- | -: | ------: | --------------: | --------: | ------: | ---: |",
        ]
    )
    for rank, row in enumerate(onsite, start=1):
        lines.append(
            f"| {rank} | {row['strategy']} | {row['N']} | {row['num_origins']} | "
            f"{row['contact_closure_max_abs']:.8g} | {row['bare_total_ward_max_abs']:.8g} | "
            f"{row['amplitude_phase_ward_max_abs']:.8g} | {row['num_k_points_total']} |"
        )

    lines.extend(
        [
            "",
            "## spm/dwave monitor",
            "",
            "| pairing | strategy | N | q | AP Ward | sigma diag min | offdiag rel | max sigma tilde |",
            "| ------- | -------- | -: | - | ------: | -------------: | ----------: | --------------: |",
        ]
    )
    for row in payload["cases"]:
        if row["pairing"] == "onsite_s":
            continue
        lines.append(
            f"| {row['pairing']} | {row['strategy']} | {row['N']} | {_q_label(row)} | "
            f"{row['amplitude_phase_ward_max_abs']:.8g} | {row['sigma_diag_min_real']:.8g} | "
            f"{row['sigma_offdiag_rel']:.8g} | {row['max_abs_sigma_tilde']:.8g} |"
        )

    recommendation = payload["recommended_strategy_for_stageSC_2b"]
    lines.extend(
        [
            "",
            "## Recommended strategy",
            "",
            "| recommended strategy | N | origins | reason |",
            "| -------------------- | -: | ------: | ------ |",
        ]
    )
    if recommendation is None:
        lines.append(f"| none | - | - | {payload['no_recommendation_reason']} |")
    else:
        lines.append(
            f"| {recommendation['strategy']} | {recommendation['N']} | "
            f"{recommendation['num_origins']} | {recommendation['reason']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", dest="quick", action="store_true", default=True)
    mode.add_argument("--full", dest="quick", action="store_false")
    args = parser.parse_args()
    payload = build_payload(args.quick)
    _write_report(payload)


if __name__ == "__main__":
    main()
