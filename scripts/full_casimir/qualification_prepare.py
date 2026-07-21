from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import DEFAULT_OUTPUT_ROOT, case_name, select_runtime_resources, validate_pairings
from .data_management import _digest, _sha, _write
from .qualification import (
    HOLDOUT_PLAN_SCHEMA,
    HOLDOUT_SAFETY_FACTOR,
    LOGDET_RTOL,
    PROFILE,
    SOURCE_PROFILE,
    _audit_holdout_rows,
    _cache_entry_map,
    _full_config,
    _holdout_item,
    _local_uncertainty,
    _projection_one,
    _read_audit,
)


def _identity(row: Mapping[str, Any]) -> tuple[str, int, str, str] | None:
    value = row.get("identity")
    if not isinstance(value, list) or len(value) != 4:
        return None
    return str(value[0]), int(value[1]), str(value[2]), str(value[3])


def select_holdout_keys(
    *,
    entries: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    weighted: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    projection_reports: Sequence[Mapping[str, Any]],
    max_points: int,
) -> tuple[list[tuple[str, int, str, str]], dict[tuple[str, int, str, str], set[str]]]:
    """Select a bounded independent holdout without treating every earlier stop as mandatory."""

    if max_points <= 0:
        raise ValueError("max_points must be positive")
    reasons: dict[tuple[str, int, str, str], set[str]] = {}
    mandatory: set[tuple[str, int, str, str]] = set()
    for report in projection_reports:
        for raw in report.get("decisions", []):
            if not isinstance(raw, Mapping):
                continue
            key = _identity(raw)
            if key is None or key not in entries:
                continue
            source_status = str(raw.get("source_status"))
            target_status = str(raw.get("target_status"))
            if source_status != target_status and target_status == "established":
                mandatory.add(key)
                reasons.setdefault(key, set()).add(
                    "acceptance_status_changed_under_frozen_candidate"
                )
            elif bool(raw.get("decision_changed")):
                reasons.setdefault(key, set()).add("candidate_stops_at_different_N")

    if len(mandatory) > max_points:
        raise ValueError(
            f"mandatory acceptance-boundary holdout count {len(mandatory)} exceeds "
            f"max_points={max_points}; raise the explicit cap rather than silently truncating"
        )
    selected = sorted(mandatory)

    ranked = sorted(
        (key for key in weighted if key in entries and key not in mandatory),
        key=lambda key: float(weighted[key].get("weighted_error_contribution_J_m2") or 0.0),
        reverse=True,
    )
    control_slots = min(2, max_points - len(selected))
    weighted_limit = max_points - control_slots
    for key in ranked:
        if len(selected) >= weighted_limit:
            break
        selected.append(key)
        reasons.setdefault(key, set()).add("largest_quadrature_weighted_uncertainty")

    for pairing in ("spm", "dwave"):
        if len(selected) >= max_points:
            break
        controls: list[tuple[float, tuple[str, int, str, str]]] = []
        for key, entry in entries.items():
            if key[0] != pairing or key in selected:
                continue
            try:
                uncertainty = _local_uncertainty(entry["point_result"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(uncertainty):
                controls.append((uncertainty, key))
        if controls:
            _, key = min(controls, key=lambda value: value[0])
            selected.append(key)
            reasons.setdefault(key, set()).add("easy_pairing_control")

    for key in ranked:
        if len(selected) >= max_points:
            break
        if key not in selected:
            selected.append(key)
            reasons.setdefault(key, set()).add("largest_quadrature_weighted_uncertainty")

    if not selected:
        raise ValueError("qualification holdout selection is empty")
    if len(selected) > max_points:
        raise AssertionError("qualification holdout selection exceeded its explicit cap")
    return selected, reasons


def build_bounded_holdout_plan(
    *,
    audit_path: Path,
    output_root: Path,
    source_profile: str,
    target_profile: str,
    projection_reports: Sequence[Mapping[str, Any]],
    max_points: int,
) -> dict[str, Any]:
    audit = _read_audit(audit_path)
    weighted = _audit_holdout_rows(audit)
    entries: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    target_cache_sha: dict[str, str] = {}
    for pairing in ("spm", "dwave"):
        target_run = Path(output_root) / case_name(pairing, 0, profile=target_profile)
        cache_path = target_run / "cache" / "certified_points.json"
        entries.update(_cache_entry_map(cache_path))
        target_cache_sha[pairing] = _sha(cache_path)

    selected, reasons = select_holdout_keys(
        entries=entries,
        weighted=weighted,
        projection_reports=projection_reports,
        max_points=int(max_points),
    )
    items = [
        _holdout_item(
            key,
            entries[key],
            reasons=sorted(reasons.get(key, {"stratified_control"})),
            weighted_row=weighted.get(key),
        )
        for key in selected
    ]
    payload = {
        "schema": HOLDOUT_PLAN_SCHEMA,
        "profile": target_profile,
        "source_profile": source_profile,
        "candidate_logdet_rtol": LOGDET_RTOL,
        "safety_factor": HOLDOUT_SAFETY_FACTOR,
        "audit_report": str(Path(audit_path).resolve()),
        "audit_value_sha256": _digest(audit),
        "source_artifact_sha256": {
            str(report["pairing"]): dict(report["source_artifact_sha256"])
            for report in projection_reports
        },
        "target_cache_sha256": target_cache_sha,
        "selection_count": len(items),
        "max_primary_points": int(max_points),
        "mandatory_acceptance_boundary_count": sum(
            "acceptance_status_changed_under_frozen_candidate" in reasons.get(key, set())
            for key in selected
        ),
        "items": items,
        "independence_contract": {
            "candidate_frozen_before_execution": True,
            "results_must_not_retune_candidate": True,
            "two_predeclared_N_levels": True,
            "every_selected_point_must_pass": True,
            "selection_cap_enforced": True,
        },
    }
    payload["plan_sha256"] = _digest(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.qualification_prepare",
        description="Project immutable v4 histories and write a bounded v5 holdout plan.",
    )
    parser.add_argument("--pairings", nargs="+", default=("spm", "dwave"), choices=("spm", "dwave"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-profile", default=SOURCE_PROFILE)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=Path("outputs/casimir/reports/convergence_audit.compact.json"),
    )
    parser.add_argument(
        "--holdout-plan",
        type=Path,
        default=Path("outputs/casimir/catalog/0deg_qualification_v5_holdout_plan.json"),
    )
    parser.add_argument("--max-holdout-points", type=int, default=32)
    parser.add_argument("--reserve-cpus", type=int, default=6)
    parser.add_argument("--worker-cap", type=int, default=26)
    parser.add_argument("--memory-budget-gb", type=float, default=16.0)
    parser.add_argument("--max-context-workers", type=int, default=1)
    parser.add_argument("--parallel-mode", choices=("auto", "serial", "q", "context", "wave"), default="q")
    parser.add_argument("--certifier-q-batch-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        pairings = validate_pairings(args.pairings)
        if set(pairings) != {"spm", "dwave"}:
            raise ValueError("qualification preparation requires both spm and dwave")
        resources = select_runtime_resources(
            reserve_logical_cpus=int(args.reserve_cpus),
            worker_cap=int(args.worker_cap),
        )
        reports = []
        for pairing in pairings:
            target_run = Path(args.output_root) / case_name(pairing, 0, profile=str(args.profile))
            target_config = _full_config(
                pairing,
                cache_path=target_run / "cache" / "certified_points.json",
                resources=resources,
                memory_budget_gb=float(args.memory_budget_gb),
                max_context_workers=int(args.max_context_workers),
                parallel_mode=str(args.parallel_mode),
                certifier_q_batch_size=int(args.certifier_q_batch_size),
            )
            report = _projection_one(
                pairing=pairing,
                output_root=Path(args.output_root),
                source_profile=str(args.source_profile),
                target_profile=str(args.profile),
                target_config=target_config,
            )
            reports.append(report)
            print(
                f"projection {pairing}: retained={report['retained_entry_count']} "
                f"omitted={report['omitted_entry_count']} "
                f"changed={report['changed_decision_count']}",
                flush=True,
            )
        plan = build_bounded_holdout_plan(
            audit_path=Path(args.audit_report),
            output_root=Path(args.output_root),
            source_profile=str(args.source_profile),
            target_profile=str(args.profile),
            projection_reports=reports,
            max_points=int(args.max_holdout_points),
        )
        _write(Path(args.holdout_plan), plan)
        print(f"written: {Path(args.holdout_plan).resolve()}")
        print(f"holdout points: {plan['selection_count']}")
        print(f"mandatory boundary points: {plan['mandatory_acceptance_boundary_count']}")
        print(f"plan_sha256: {plan['plan_sha256']}")
        print(f"selected CPUs: {resources.selected_cpus}")
        print("No microscopic holdout work was executed.")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"QUALIFICATION PREPARE FAILED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
