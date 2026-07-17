"""Total Matsubara staged scan with observable-aware static and positive soft gates.

This command reuses the established single-method composite-Gauss orchestration and
microscopic backend.  It changes only the acceptance policy: exact ``n=0`` remains on
the strict static physics path, while its numerical primary-response drift may be
accepted at a separately reported soft threshold when observables converge, the trend
is non-worsening, and a shifted-periodic-cut audit passes.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

from validation.commands.matsubara import positive_orbit_gauss_scan as _base


@dataclass(frozen=True)
class StepMetrics:
    physical_all: bool
    observable_all: bool
    static_strict_all: bool
    static_soft_all: bool
    positive_strict_all: bool
    positive_soft_all: bool
    strict_all: bool
    soft_all: bool
    max_primary_relative: float
    max_static_relative: float
    max_positive_relative: float
    max_reflection_relative: float
    max_logdet_relative: float
    rows: tuple[dict[str, Any], ...]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    policy = argparse.ArgumentParser(add_help=False)
    policy.add_argument("--strict-static-rtol", type=float, default=None)
    policy.add_argument("--soft-static-rtol", type=float, default=None)
    policy_args, remaining = policy.parse_known_args(raw)
    args = _base._parse_args(remaining)
    args.strict_static_rtol = float(
        args.strict_sigma_rtol
        if policy_args.strict_static_rtol is None
        else policy_args.strict_static_rtol
    )
    args.soft_static_rtol = float(
        args.soft_sigma_rtol
        if policy_args.soft_static_rtol is None
        else policy_args.soft_static_rtol
    )
    if not (
        math.isfinite(args.strict_static_rtol)
        and math.isfinite(args.soft_static_rtol)
        and 0.0 <= args.strict_static_rtol <= args.soft_static_rtol
    ):
        raise SystemExit(
            "require finite 0 <= --strict-static-rtol <= --soft-static-rtol"
        )
    return args


def _metrics(
    path: Path,
    *,
    static_strict: float,
    static_soft: float,
    sigma_strict: float,
    sigma_soft: float,
    observable: float,
) -> StepMetrics:
    rows = _base._read_rows(path)
    if not rows:
        raise ValueError(f"no rows in {path}")
    physical_all = all(
        _base._as_bool(row["point_pipeline_passed"])
        and _base._as_bool(row["ward_passed"])
        and _base._as_bool(row["sheet_validation_passed"])
        and _base._as_bool(row["reflection_constructed"])
        and _base._as_bool(row["logdet_passed"])
        for row in rows
    )
    primary_values = tuple(
        _base._finite_float(row["reference_primary_response_relative"])
        for row in rows
    )
    static_values = tuple(
        _base._finite_float(row["reference_primary_response_relative"])
        for row in rows
        if int(row["matsubara_index"]) == 0
    )
    positive_values = tuple(
        _base._finite_float(row["reference_primary_response_relative"])
        for row in rows
        if int(row["matsubara_index"]) > 0
    )
    reflection_values = tuple(
        _base._finite_float(row["reference_reflection_matrix_relative"])
        for row in rows
    )
    logdet_values = tuple(
        _base._finite_float(row["reference_logdet_relative"]) for row in rows
    )
    max_primary = max(primary_values, default=float("inf"))
    max_static = max(static_values, default=float("inf"))
    max_positive = max(positive_values, default=float("inf"))
    max_reflection = max(reflection_values, default=float("inf"))
    max_logdet = max(logdet_values, default=float("inf"))
    zero_rows = [row for row in rows if int(row["matsubara_index"]) == 0]
    static_physics = bool(
        len(zero_rows) == 1
        and _base._as_bool(zero_rows[0]["strict_static_ward_passed"])
    )
    static_strict_all = bool(static_physics and max_static <= static_strict)
    static_soft_all = bool(static_physics and max_static <= static_soft)
    positive_strict_all = bool(
        positive_values and max_positive <= sigma_strict
    )
    positive_soft_all = bool(positive_values and max_positive <= sigma_soft)
    observable_all = bool(
        physical_all
        and max_reflection <= observable
        and max_logdet <= observable
    )
    return StepMetrics(
        physical_all=physical_all,
        observable_all=observable_all,
        static_strict_all=static_strict_all,
        static_soft_all=static_soft_all,
        positive_strict_all=positive_strict_all,
        positive_soft_all=positive_soft_all,
        strict_all=bool(
            observable_all and static_strict_all and positive_strict_all
        ),
        soft_all=bool(observable_all and static_soft_all and positive_soft_all),
        max_primary_relative=max_primary,
        max_static_relative=max_static,
        max_positive_relative=max_positive,
        max_reflection_relative=max_reflection,
        max_logdet_relative=max_logdet,
        rows=rows,
    )


def _sector_acceptance(metrics: StepMetrics) -> tuple[str, str]:
    static = (
        "strict"
        if metrics.static_strict_all
        else "soft"
        if metrics.static_soft_all
        else "unresolved"
    )
    positive = (
        "strict"
        if metrics.positive_strict_all
        else "soft"
        if metrics.positive_soft_all
        else "unresolved"
    )
    return static, positive


def _final_rows(
    *,
    pairing: str,
    case: _base.CaseSpec,
    reference_order: int,
    final_order: int,
    classification: str,
    metrics: StepMetrics,
    cut_audit: StepMetrics | None,
) -> list[dict[str, Any]]:
    static_acceptance, positive_acceptance = _sector_acceptance(metrics)
    result = _base._final_rows(
        pairing=pairing,
        case=case,
        reference_order=reference_order,
        final_order=final_order,
        classification=classification,
        metrics=metrics,
        cut_audit=cut_audit,
    )
    for row in result:
        row["static_acceptance"] = static_acceptance
        row["positive_acceptance"] = positive_acceptance
        row["static_soft_all"] = metrics.static_soft_all
        row["cut_audit_soft_passed"] = bool(
            cut_audit is not None and cut_audit.soft_all
        )
    return result


def main() -> None:
    args = _parse_args()
    preflight_payload: dict[str, Any] | None = None
    if args.require_preflight and not args.dry_run:
        preflight_payload = _base._validate_preflight(args)

    print("single-method total Matsubara orbit staged scan", flush=True)
    print(
        f"pairings={args.pairings}; cases={[case.label for case in args.cases]}",
        flush=True,
    )
    print(
        f"Gauss stage pairs={args.gauss_stages}; Matsubara n={args.matsubara_indices}",
        flush=True,
    )
    print(
        "static rtol strict/soft="
        f"{args.strict_static_rtol:.3e}/{args.soft_static_rtol:.3e}; "
        "positive rtol strict/soft="
        f"{args.strict_sigma_rtol:.3e}/{args.soft_sigma_rtol:.3e}",
        flush=True,
    )
    print(
        "exact n=0 physics remains hard; only numerical response drift may soft-pass",
        flush=True,
    )
    if preflight_payload is not None:
        print(
            f"preflight accepted for git head {preflight_payload.get('git_head')}",
            flush=True,
        )

    if args.dry_run:
        for pairing in args.pairings:
            for case in args.cases:
                print(f"would scan {pairing}/{case.label} through {args.gauss_stages}")
        return

    final_rows: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for pairing in args.pairings:
        for case in args.cases:
            case_root = args.output_root / "raw" / pairing / case.label
            previous_stage_static = float("inf")
            previous_stage_positive = float("inf")
            soft_streak = 0
            classification = "unresolved"
            final_reference_order = int(args.gauss_stages[-1][0])
            final_order = int(args.gauss_stages[-1][1])
            final_metrics: StepMetrics | None = None

            for stage_index, (low_order, high_order) in enumerate(args.gauss_stages):
                low_output = case_root / f"c{low_order}.csv"
                high_output = case_root / f"c{high_order}.csv"
                _base._run_command(
                    args=args,
                    pairing=pairing,
                    case=case,
                    order=low_order,
                    output=low_output,
                    reference_csv=None,
                    stage=f"stage_{stage_index}_low",
                )
                _base._run_command(
                    args=args,
                    pairing=pairing,
                    case=case,
                    order=high_order,
                    output=high_output,
                    reference_csv=low_output,
                    stage=f"stage_{stage_index}_high",
                )
                metrics = _metrics(
                    high_output,
                    static_strict=args.strict_static_rtol,
                    static_soft=args.soft_static_rtol,
                    sigma_strict=args.strict_sigma_rtol,
                    sigma_soft=args.soft_sigma_rtol,
                    observable=args.observable_rtol,
                )
                final_metrics = metrics
                final_reference_order = low_order
                final_order = high_order

                if metrics.strict_all:
                    classification = "strict"
                    break

                nonworsening = bool(
                    metrics.max_static_relative
                    <= previous_stage_static * (1.0 + 1e-12)
                    and metrics.max_positive_relative
                    <= previous_stage_positive * (1.0 + 1e-12)
                )
                if metrics.soft_all and nonworsening:
                    soft_streak += 1
                elif metrics.soft_all:
                    soft_streak = 1
                else:
                    soft_streak = 0
                previous_stage_static = metrics.max_static_relative
                previous_stage_positive = metrics.max_positive_relative
                if metrics.soft_all and soft_streak >= args.soft_confirmations:
                    classification = "soft_confirmed"
                    break

            if final_metrics is None:
                raise RuntimeError(
                    f"no comparable order pair for {pairing}/{case.label}"
                )
            if classification == "unresolved":
                if final_metrics.soft_all:
                    classification = "soft_at_max_order"
                elif not final_metrics.physical_all:
                    classification = "physical_failure"
                elif not final_metrics.observable_all:
                    classification = "observable_failure"
                elif not final_metrics.static_soft_all:
                    classification = "static_response_unresolved"
                elif not final_metrics.positive_soft_all:
                    classification = "positive_response_unresolved"

            cut_metrics: StepMetrics | None = None
            if args.soft_cut_audit and classification.startswith("soft"):
                final_csv = case_root / f"c{final_order}.csv"
                audit_output = case_root / f"c{final_order}_shifted_cut.csv"
                _base._run_command(
                    args=args,
                    pairing=pairing,
                    case=case,
                    order=final_order,
                    output=audit_output,
                    reference_csv=final_csv,
                    integration_start=-np.pi + args.cut_audit_shift,
                    stage="soft_cut_audit",
                )
                cut_metrics = _metrics(
                    audit_output,
                    static_strict=args.strict_static_rtol,
                    static_soft=args.soft_static_rtol,
                    sigma_strict=args.strict_sigma_rtol,
                    sigma_soft=args.soft_sigma_rtol,
                    observable=args.observable_rtol,
                )
                if not cut_metrics.soft_all:
                    classification = "cut_audit_failure"

            static_acceptance, positive_acceptance = _sector_acceptance(final_metrics)
            final_rows.extend(
                _final_rows(
                    pairing=pairing,
                    case=case,
                    reference_order=final_reference_order,
                    final_order=final_order,
                    classification=classification,
                    metrics=final_metrics,
                    cut_audit=cut_metrics,
                )
            )
            case_records.append(
                {
                    "pairing": pairing,
                    "case": case.label,
                    "mx": case.mx,
                    "my": case.my,
                    "classification": classification,
                    "static_acceptance": static_acceptance,
                    "positive_acceptance": positive_acceptance,
                    "reference_gauss_order": final_reference_order,
                    "final_gauss_order": final_order,
                    "max_primary_relative": final_metrics.max_primary_relative,
                    "max_static_relative": final_metrics.max_static_relative,
                    "max_positive_relative": final_metrics.max_positive_relative,
                    "max_reflection_relative": final_metrics.max_reflection_relative,
                    "max_logdet_relative": final_metrics.max_logdet_relative,
                    "physical_all": final_metrics.physical_all,
                    "observable_all": final_metrics.observable_all,
                    "static_strict_all": final_metrics.static_strict_all,
                    "static_soft_all": final_metrics.static_soft_all,
                    "positive_strict_all": final_metrics.positive_strict_all,
                    "positive_soft_all": final_metrics.positive_soft_all,
                    "strict_all": final_metrics.strict_all,
                    "soft_all": final_metrics.soft_all,
                    "cut_audit_performed": cut_metrics is not None,
                    "cut_audit_soft_passed": bool(
                        cut_metrics is not None and cut_metrics.soft_all
                    ),
                }
            )
            print(
                f"[final] {pairing}/{case.label}: {classification}; "
                f"C{final_reference_order}/C{final_order}; "
                f"static={final_metrics.max_static_relative:.3e}({static_acceptance}); "
                f"positive={final_metrics.max_positive_relative:.3e}({positive_acceptance}); "
                f"R={final_metrics.max_reflection_relative:.3e}; "
                f"logdet={final_metrics.max_logdet_relative:.3e}",
                flush=True,
            )
            if args.stop_on_error and classification in {
                "physical_failure",
                "static_response_unresolved",
                "positive_response_unresolved",
                "observable_failure",
                "cut_audit_failure",
            }:
                raise SystemExit(
                    f"stopping on {pairing}/{case.label}: {classification}"
                )

    if not final_rows or not case_records:
        raise RuntimeError("staged scan produced no final rows")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    points_csv = output_root / "scan_points.csv"
    with points_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    cases_csv = output_root / "scan_cases.csv"
    with cases_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_records[0]))
        writer.writeheader()
        writer.writerows(case_records)

    accepted_classes = {"strict", "soft_confirmed", "soft_at_max_order"}
    total_points = len(final_rows)
    strict_points = sum(
        row["classification"] == "strict"
        and float(row["primary_response_relative"])
        <= (
            args.strict_static_rtol
            if int(row["matsubara_index"]) == 0
            else args.strict_sigma_rtol
        )
        for row in final_rows
    )
    accepted_points = sum(
        row["classification"] in accepted_classes
        and float(row["primary_response_relative"])
        <= (
            args.soft_static_rtol
            if int(row["matsubara_index"]) == 0
            else args.soft_sigma_rtol
        )
        and float(row["reflection_relative"]) <= args.observable_rtol
        and float(row["logdet_relative"]) <= args.observable_rtol
        for row in final_rows
    )
    zero_rows = [row for row in final_rows if int(row["matsubara_index"]) == 0]
    strict_fraction = strict_points / total_points
    accepted_fraction = accepted_points / total_points
    all_closure = all(bool(row["point_pipeline_passed"]) for row in final_rows)
    all_observables = all(
        float(row["reflection_relative"]) <= args.observable_rtol
        and float(row["logdet_relative"]) <= args.observable_rtol
        for row in final_rows
    )
    all_static_strict = bool(zero_rows) and all(
        bool(row["strict_static_ward_passed"])
        and float(row["primary_response_relative"]) <= args.strict_static_rtol
        for row in zero_rows
    )
    all_static_accepted = bool(zero_rows) and all(
        bool(row["strict_static_ward_passed"])
        and float(row["primary_response_relative"]) <= args.soft_static_rtol
        for row in zero_rows
    )
    all_cases_accepted = all(
        record["classification"] in accepted_classes for record in case_records
    )
    outer_candidate = bool(
        preflight_payload is not None
        and all_closure
        and all_observables
        and all_static_accepted
        and all_cases_accepted
        and accepted_fraction == 1.0
        and strict_fraction >= args.minimum_strict_fraction
    )
    total_wall = float(time.perf_counter() - total_started)
    payload = {
        "schema": "total_matsubara_pointwise_gauss_scan_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _base._git_head(),
        "arguments": {
            key: _base._jsonable(value) for key, value in vars(args).items()
        },
        "preflight": {
            "manifest": str(args.preflight_manifest),
            "required": bool(args.require_preflight),
            "accepted": preflight_payload is not None,
            "git_head": (
                None
                if preflight_payload is None
                else preflight_payload.get("git_head")
            ),
        },
        "total_wall_seconds": total_wall,
        "cases": case_records,
        "status": {
            "single_transverse_method": (
                "full_period_equal_panel_composite_gauss_legendre"
            ),
            "gauss_stage_pairs": [list(stage) for stage in args.gauss_stages],
            "zero_matsubara_included": bool(zero_rows),
            "zero_uses_exact_static_divided_difference": True,
            "zero_conductivity_division_used": False,
            "zero_and_positive_share_eigensystems": True,
            "static_physics_contract_softened": False,
            "static_numerical_soft_acceptance_enabled": True,
            "all_closure_checks_passed": all_closure,
            "all_observable_checks_passed": all_observables,
            "all_static_points_strict": all_static_strict,
            "all_static_points_strict_or_soft": all_static_accepted,
            "strict_point_fraction": strict_fraction,
            "strict_or_soft_point_fraction": accepted_fraction,
            "minimum_strict_fraction": args.minimum_strict_fraction,
            "all_cases_accepted": all_cases_accepted,
            "preflight_passed": preflight_payload is not None,
            "outer_integral_candidate": outer_candidate,
            "diagnostic_only": True,
            "production_reference_established": False,
            "valid_for_casimir_input": False,
        },
    }
    summary_json = output_root / "scan_summary.json"
    summary_json.write_text(
        json.dumps(_base._jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"scan complete: closure={all_closure}, "
        f"observables={all_observables}, "
        f"static_strict={all_static_strict}, "
        f"static_accepted={all_static_accepted}, "
        f"strict_fraction={strict_fraction:.3f}, "
        f"accepted_fraction={accepted_fraction:.3f}, "
        f"outer_integral_candidate={outer_candidate}",
        flush=True,
    )
    print(f"points CSV: {points_csv}")
    print(f"cases CSV:  {cases_csv}")
    print(f"summary:    {summary_json}")


if __name__ == "__main__":
    main()
