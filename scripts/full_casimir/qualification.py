from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from lno327.casimir.cli import execute_case
from lno327.casimir.certified_point_provider import (
    certified_point_policy_fingerprint,
    certified_point_policy_payload,
)
from lno327.casimir.fixed_chain import FixedCasimirConfig
from lno327.casimir.fixed_outer_q import OuterQNodeManifest
from lno327.casimir.fixed_transverse_point_certification import (
    ENVELOPE_LEVELS,
    assess_frequency_level,
    assess_oscillatory_envelope,
)
from lno327.casimir.production import build_full_casimir_config
from lno327.casimir.qualification import run_qualification_casimir
from lno327.casimir.strict_transverse_runner import run_strict_transverse_certifier

from .cache_migration import (
    CACHE_SCHEMA,
    _atomic_json,
    _point_config_from_run_config,
    _read_json_mapping,
    _validated_entries,
)
from .config import (
    DEFAULT_OUTPUT_ROOT,
    RuntimeResources,
    case_name,
    select_runtime_resources,
    validate_pairings,
)
from .data_management import _digest, _read, _sha, _write
from .data_retention import _unpack_value
from .policy_audit import compare_policy_snapshots

PROFILE = "0deg_qualification_v5"
SOURCE_PROFILE = "0deg_pilot_v4"
LOGDET_RTOL = 2.0e-3
LOGDET_ATOL = 1.0e-6
REQUIRED_CONSECUTIVE_PASSES = 2
N_CANDIDATES = (128, 192, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280)
RADIAL_BUDGET_FRACTION = 0.8
OUTER_CUTOFFS_U = (6.0, 10.0, 14.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0)
MATSUBARA_CUTOFFS = (1, 3, 7, 11, 15, 23, 31)
TOTAL_RTOL = 5.0e-3
TOTAL_ATOL_J_M2 = 1.0e-12
HOLDOUT_SAFETY_FACTOR = 2.0
PROJECTION_SCHEMA = "zero-degree-qualification-cache-projection-v1"
HOLDOUT_PLAN_SCHEMA = "zero-degree-qualification-holdout-plan-v1"
HOLDOUT_EXECUTION_SCHEMA = "zero-degree-qualification-holdout-execution-v1"
PREFLIGHT_SCHEMA = "zero-degree-qualification-preflight-v1"

_REQUIRED_SOURCE_ARTIFACTS = (
    "config.json",
    "manifest.json",
    "result.json",
    "summary.json",
    "cache/certified_points.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise RuntimeError("cannot resolve the qualification Git commit")
    return value


def _require_clean_git() -> None:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(_repo_root()),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot inspect the qualification Git worktree")
    if completed.stdout.strip():
        raise RuntimeError("qualification preflight requires a clean tracked worktree")


def _source_hashes(run_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in _REQUIRED_SOURCE_ARTIFACTS:
        path = Path(run_dir) / relative
        if not path.is_file():
            raise FileNotFoundError(f"qualification source artifact is missing: {path}")
        output[relative] = _sha(path)
    return output


def _full_config(
    pairing: str,
    *,
    cache_path: Path,
    resources: RuntimeResources,
    memory_budget_gb: float,
    max_context_workers: int,
    parallel_mode: str,
    certifier_q_batch_size: int,
):
    return build_full_casimir_config(
        point_cache_path=cache_path,
        pairings=(pairing,),
        temperature_K=10.0,
        separation_nm=20.0,
        plate_angles_deg=(0.0, 0.0),
        delta0_eV=0.1,
        eta_eV=1e-8,
        degeneracy=1.0,
        N_candidates=N_CANDIDATES,
        required_consecutive_passes=REQUIRED_CONSECUTIVE_PASSES,
        logdet_rtol=LOGDET_RTOL,
        logdet_atol=LOGDET_ATOL,
        workers=resources.workers,
        parallel_mode=parallel_mode,
        memory_budget_gb=float(memory_budget_gb),
        max_context_workers=int(max_context_workers),
        cutoff_u_values=OUTER_CUTOFFS_U,
        outer_tail_start_u=24.0,
        outer_tail_window_shells=3,
        outer_tail_ratio_max=0.8,
        matsubara_cutoff_values=MATSUBARA_CUTOFFS,
        matsubara_tail_start_n=8,
        matsubara_tail_window_terms=4,
        matsubara_tail_ratio_max=0.8,
        total_free_energy_rtol=TOTAL_RTOL,
        total_free_energy_atol_J_m2=TOTAL_ATOL_J_M2,
        radial_budget_fraction=RADIAL_BUDGET_FRACTION,
        certifier_q_batch_size=int(certifier_q_batch_size),
    )


def _assert_projection_compatible(
    source: FixedCasimirConfig,
    target: FixedCasimirConfig,
) -> None:
    left = certified_point_policy_payload(source, frequency_extendable=True)
    right = certified_point_policy_payload(target, frequency_extendable=True)
    source_rtol = float(left.pop("logdet_rtol"))
    target_rtol = float(right.pop("logdet_rtol"))
    source_N = tuple(int(value) for value in left.pop("N_candidates"))
    target_N = tuple(int(value) for value in right.pop("N_candidates"))
    if left != right:
        raise ValueError(
            "qualification projection permits only the frozen relative-tolerance "
            "relaxation and a strict N-ladder prefix extension"
        )
    if target_rtol < source_rtol:
        raise ValueError("qualification target logdet_rtol is stricter than its source")
    if target_N != source_N and (
        len(target_N) <= len(source_N) or target_N[: len(source_N)] != source_N
    ):
        raise ValueError("qualification target N ladder is not a source-prefix extension")


def _reassess_complete_history(
    point: Mapping[str, Any],
    *,
    rtol: float,
    atol: float,
    required_consecutive_passes: int,
) -> dict[str, Any]:
    output = deepcopy(dict(point))
    rebuilt: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any]] | None = None
    consecutive = 0
    established: dict[str, Any] | None = None
    for raw_row in point.get("history", []):
        if not isinstance(raw_row, Mapping):
            raise ValueError("point history contains a malformed row")
        row = deepcopy(dict(raw_row))
        shifts = row.get("shifts")
        if not isinstance(shifts, Mapping) or not shifts:
            raise ValueError("point history row has no shift states")
        current = {str(key): deepcopy(dict(value)) for key, value in shifts.items()}
        assessment = assess_frequency_level(
            current_by_shift=current,
            previous_by_shift=previous,
            rtol=float(rtol),
            atol=float(atol),
        )
        consecutive = consecutive + 1 if assessment["accepted_transition"] else 0
        row["shifts"] = current
        row.update(assessment)
        row["consecutive_accepted_transitions"] = consecutive
        rebuilt.append(row)
        envelope = assess_oscillatory_envelope(
            rebuilt,
            rtol=float(rtol),
            atol=float(atol),
        )
        row["oscillatory_envelope"] = envelope
        strict_ready = consecutive >= int(required_consecutive_passes)
        envelope_ready = bool(envelope["passed"])
        if established is None and (strict_ready or envelope_ready):
            if len(rebuilt) < 2:
                raise ValueError("reassessed convergence has no previous N level")
            established = {
                "status": "established",
                "establishment_mode": (
                    "strict_consecutive_adjacent"
                    if strict_ready
                    else "three_level_oscillatory_envelope"
                ),
                "working_N": int(rebuilt[-2]["N"]),
                "audit_N": int(rebuilt[-1]["N"]),
                "required_consecutive_passes": int(required_consecutive_passes),
                "envelope_levels": int(ENVELOPE_LEVELS),
                "envelope_N_window": list(envelope["N_window"]),
                "criterion": (
                    "universal hard physical closure and cross-shift stability plus "
                    "either consecutive adjacent-N convergence or a three-level "
                    "absolute-first, relative-fallback oscillatory envelope"
                ),
            }
        previous = current
    output["history"] = rebuilt
    output["sweet_spot"] = established or {
        "status": "not_established",
        "establishment_mode": None,
        "working_N": None,
        "audit_N": None,
    }
    return output


def _annotate_power_metric_certificate(point: dict[str, Any], *, n: int) -> None:
    for row in point.get("history", []):
        for state in row.get("shifts", {}).values():
            for plate_name in ("plate_1", "plate_2"):
                plate = state.get(plate_name)
                if not isinstance(plate, dict):
                    continue
                if "power_metric_contraction_certified" in plate:
                    continue
                constructed = bool(plate.get("reflection_constructed"))
                sheet_passed = bool(plate.get("sheet_validation_passed"))
                if int(n) == 0:
                    try:
                        upper = abs(float(plate.get("reflection_norm")))
                    except (TypeError, ValueError, OverflowError):
                        upper = math.inf
                    certified = bool(
                        constructed and sheet_passed and math.isfinite(upper) and upper <= 1.0 + 1e-12
                    )
                    method = "stored_frobenius_norm_upper_bounds_static_spectral_norm"
                else:
                    upper = 1.0
                    certified = bool(constructed and sheet_passed)
                    method = "passive_sheet_vacuum_admittance_similarity_theorem"
                plate["power_metric_contraction_certified"] = certified
                plate["power_metric_singular_value_max_upper_bound"] = upper
                plate["power_metric_certificate_method"] = method


def _accepted_row(point: Mapping[str, Any]) -> Mapping[str, Any]:
    sweet = point.get("sweet_spot", {})
    if not isinstance(sweet, Mapping) or sweet.get("status") != "established":
        raise ValueError("qualification point is not established")
    audit_N = int(sweet["audit_N"])
    for row in point.get("history", []):
        if isinstance(row, Mapping) and int(row.get("N", -1)) == audit_N:
            return row
    raise ValueError("qualification point history does not contain audit_N")


def _point_power_metric_certified(point: Mapping[str, Any]) -> bool:
    row = _accepted_row(point)
    shifts = row.get("shifts")
    if not isinstance(shifts, Mapping) or not shifts:
        return False
    for state in shifts.values():
        if not isinstance(state, Mapping) or not bool(state.get("hard_physical_passed")):
            return False
        for plate_name in ("plate_1", "plate_2"):
            plate = state.get(plate_name)
            if not isinstance(plate, Mapping) or not bool(
                plate.get("power_metric_contraction_certified")
            ):
                return False
    return True


def _local_uncertainty(point: Mapping[str, Any]) -> float:
    row = _accepted_row(point)
    candidates: list[float] = []
    cross = row.get("two_plate_logdet_cross_shift")
    if isinstance(cross, Mapping) and isinstance(cross.get("absolute"), (int, float)):
        candidates.append(abs(float(cross["absolute"])))
    adjacent = row.get("adjacent_N_by_shift")
    if isinstance(adjacent, Mapping):
        for record in adjacent.values():
            if isinstance(record, Mapping) and isinstance(record.get("absolute"), (int, float)):
                candidates.append(abs(float(record["absolute"])))
    envelope = row.get("oscillatory_envelope")
    if isinstance(envelope, Mapping):
        joint = envelope.get("joint_logdet_envelope")
        if isinstance(joint, Mapping) and isinstance(joint.get("absolute"), (int, float)):
            candidates.append(abs(float(joint["absolute"])))
    finite = [value for value in candidates if math.isfinite(value)]
    if not finite:
        raise ValueError("qualification point has no finite local uncertainty")
    return max(finite)


def _highest_valid_N(point: Mapping[str, Any]) -> int:
    valid: list[int] = []
    for row in point.get("history", []):
        if not isinstance(row, Mapping):
            continue
        shifts = row.get("shifts")
        if not isinstance(shifts, Mapping) or not shifts:
            continue
        if all(
            isinstance(state, Mapping)
            and bool(state.get("hard_physical_passed"))
            and isinstance(state.get("two_plate_logdet"), (int, float))
            and math.isfinite(float(state["two_plate_logdet"]))
            for state in shifts.values()
        ):
            valid.append(int(row["N"]))
    if not valid:
        raise ValueError("qualification point contains no valid N level")
    return max(valid)


def _projection_one(
    *,
    pairing: str,
    output_root: Path,
    source_profile: str,
    target_profile: str,
    target_config: Any,
) -> dict[str, Any]:
    source_run = Path(output_root) / case_name(pairing, 0, profile=source_profile)
    target_run = Path(output_root) / case_name(pairing, 0, profile=target_profile)
    source_cache = source_run / "cache" / "certified_points.json"
    target_cache = target_run / "cache" / "certified_points.json"
    target_config_path = target_run / "config.json"
    report_path = target_run / "cache" / "projection_report.json"
    source_hash_before = _source_hashes(source_run)
    source_run_config = _read_json_mapping(source_run / "config.json", label="source config")
    source_point = _point_config_from_run_config(source_run_config)
    target_point = target_config.outer_tail_config.joint_config.radial_config.point_config
    _assert_projection_compatible(source_point, target_point)

    expected_config = target_config.as_dict()
    target_fingerprint = certified_point_policy_fingerprint(
        target_point, frequency_extendable=True
    )
    target_policy = certified_point_policy_payload(target_point, frequency_extendable=True)
    if target_run.exists():
        if not target_config_path.is_file() or not target_cache.is_file() or not report_path.is_file():
            raise FileExistsError(
                f"partial qualification target exists and will not be overwritten: {target_run}"
            )
        stored_config = _read_json_mapping(target_config_path, label="target config")
        if stored_config != expected_config:
            raise ValueError(f"existing qualification target config differs: {target_run}")
        target_payload = _read_json_mapping(target_cache, label="target cache")
        _validated_entries(
            target_payload,
            path=target_cache,
            expected_fingerprint=target_fingerprint,
            expected_policy=target_policy,
        )
        report = _read_json_mapping(report_path, label="projection report")
        if report.get("schema") != PROJECTION_SCHEMA:
            raise ValueError(f"invalid existing projection report: {report_path}")
        if report.get("source_artifact_sha256") != source_hash_before:
            raise ValueError(f"source changed after qualification projection: {source_run}")
        if report.get("target_cache_sha256") != _sha(target_cache):
            raise ValueError(f"target cache changed after qualification projection: {target_cache}")
        return report

    source_payload = _read_json_mapping(source_cache, label="source cache")
    source_fingerprint = certified_point_policy_fingerprint(
        source_point, frequency_extendable=True
    )
    source_policy = certified_point_policy_payload(source_point, frequency_extendable=True)
    entries = _validated_entries(
        source_payload,
        path=source_cache,
        expected_fingerprint=source_fingerprint,
        expected_policy=source_policy,
        allow_legacy_scheduling_fingerprint=True,
    )
    retained: list[dict[str, Any]] = []
    omitted: list[list[Any]] = []
    decisions: list[dict[str, Any]] = []
    for entry in entries:
        identity = [
            str(entry["pairing"]),
            int(entry["n"]),
            str(entry["qx_hex"]),
            str(entry["qy_hex"]),
        ]
        source_status = str(
            entry["point_result"].get("sweet_spot", {}).get("status")
        )
        projected = _reassess_complete_history(
            entry["point_result"],
            rtol=LOGDET_RTOL,
            atol=LOGDET_ATOL,
            required_consecutive_passes=REQUIRED_CONSECUTIVE_PASSES,
        )
        _annotate_power_metric_certificate(projected, n=int(entry["n"]))
        target_status = str(projected.get("sweet_spot", {}).get("status"))
        changed = source_status != target_status or (
            source_status == "established"
            and entry["point_result"].get("sweet_spot") != projected.get("sweet_spot")
        )
        decisions.append(
            {
                "identity": identity,
                "source_status": source_status,
                "target_status": target_status,
                "decision_changed": changed,
                "target_working_N": projected.get("sweet_spot", {}).get("working_N"),
                "target_audit_N": projected.get("sweet_spot", {}).get("audit_N"),
            }
        )
        if target_status != "established" or not _point_power_metric_certified(projected):
            omitted.append(identity)
            continue
        projected["qualification_projection"] = {
            "schema": "qualification-point-history-projection-v1",
            "source_status": source_status,
            "target_status": target_status,
            "decision_changed": changed,
            "source_history_preserved": True,
            "frozen_logdet_rtol": LOGDET_RTOL,
            "local_absolute_uncertainty": _local_uncertainty(projected),
            "reference_highest_valid_N": _highest_valid_N(projected),
            "power_metric_contraction_certified": True,
        }
        retained.append({**entry, "point_result": projected})

    target_payload = {
        "schema": CACHE_SCHEMA,
        "policy_fingerprint": target_fingerprint,
        "frequency_extendable": True,
        "active_matsubara_indices": sorted(
            {int(entry["n"]) for entry in retained}
            or set(target_point.matsubara_indices)
        ),
        "point_policy": target_policy,
        "entries": retained,
    }
    target_run.mkdir(parents=True, exist_ok=False)
    _atomic_json(target_config_path, expected_config)
    _atomic_json(target_cache, target_payload, compact=True)
    source_hash_after = _source_hashes(source_run)
    if source_hash_before != source_hash_after:
        raise RuntimeError(f"source run changed during qualification projection: {source_run}")
    report = {
        "schema": PROJECTION_SCHEMA,
        "created_at_utc": _utc_now(),
        "pairing": pairing,
        "source_run": str(source_run.resolve()),
        "target_run": str(target_run.resolve()),
        "source_profile": source_profile,
        "target_profile": target_profile,
        "source_artifact_sha256": source_hash_before,
        "source_cache_sha256": source_hash_before["cache/certified_points.json"],
        "target_config_sha256": _sha(target_config_path),
        "target_cache_sha256": _sha(target_cache),
        "target_policy_fingerprint": target_fingerprint,
        "source_entry_count": len(entries),
        "retained_entry_count": len(retained),
        "omitted_entry_count": len(omitted),
        "omitted_identities": omitted,
        "changed_decision_count": sum(bool(row["decision_changed"]) for row in decisions),
        "decisions": decisions,
        "source_immutable": True,
        "empty_restart_forbidden": True,
    }
    report["projection_sha256"] = _digest(report)
    _atomic_json(report_path, report)
    return report


def _read_audit(path: Path) -> Mapping[str, Any]:
    payload = _read(Path(path))
    if not isinstance(payload, Mapping):
        raise ValueError("convergence audit must be a JSON object")
    restored = _unpack_value(payload)
    if not isinstance(restored, Mapping) or restored.get("schema") != "full-casimir-convergence-audit-v2":
        raise ValueError("convergence audit must use schema full-casimir-convergence-audit-v2")
    return restored


def _audit_holdout_rows(audit: Mapping[str, Any]) -> dict[tuple[str, int, str, str], Mapping[str, Any]]:
    output: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for replay in audit.get("candidate_closure_replays", []):
        if not isinstance(replay, Mapping) or not math.isclose(
            float(replay.get("candidate_logdet_rtol", -1.0)),
            LOGDET_RTOL,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            continue
        for outer in replay.get("outer_runs", []):
            if not isinstance(outer, Mapping):
                continue
            plan = outer.get("holdout_plan")
            if not isinstance(plan, Mapping):
                continue
            for row in plan.get("points", []):
                if not isinstance(row, Mapping):
                    continue
                identity = row.get("identity")
                if not isinstance(identity, list) or len(identity) != 4:
                    continue
                key = (str(identity[0]), int(identity[1]), str(identity[2]), str(identity[3]))
                output[key] = row
    return output


def _cache_entry_map(cache_path: Path) -> dict[tuple[str, int, str, str], Mapping[str, Any]]:
    payload = _read_json_mapping(cache_path, label="qualification cache")
    output: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    for raw in payload.get("entries", []):
        if not isinstance(raw, Mapping):
            continue
        key = (
            str(raw["pairing"]),
            int(raw["n"]),
            str(raw["qx_hex"]),
            str(raw["qy_hex"]),
        )
        output[key] = raw
    return output


def _holdout_levels(reference_N: int) -> tuple[int, int]:
    step = max(64, 2 * round(max(int(reference_N) // 10, 64) / 2))
    return int(reference_N + step), int(reference_N + 2 * step)


def _holdout_item(
    identity: tuple[str, int, str, str],
    entry: Mapping[str, Any],
    *,
    reasons: Sequence[str],
    weighted_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    point = entry["point_result"]
    row = _accepted_row(point)
    sweet = point["sweet_spot"]
    accepted = {
        str(label): float(state["two_plate_logdet"])
        for label, state in row["shifts"].items()
    }
    uncertainty = _local_uncertainty(point)
    if weighted_row is not None and isinstance(
        weighted_row.get("predicted_local_uncertainty"), (int, float)
    ):
        uncertainty = max(uncertainty, abs(float(weighted_row["predicted_local_uncertainty"])))
    reference_N = _highest_valid_N(point)
    holdout_N = _holdout_levels(reference_N)
    return {
        "identity": list(identity),
        "pairing": identity[0],
        "n": identity[1],
        "qx_hex": identity[2],
        "qy_hex": identity[3],
        "q_model": [float.fromhex(identity[2]), float.fromhex(identity[3])],
        "reasons": sorted(set(str(value) for value in reasons)),
        "candidate_working_N": int(sweet["working_N"]),
        "candidate_audit_N": int(sweet["audit_N"]),
        "candidate_values_by_shift": accepted,
        "predicted_local_uncertainty": uncertainty,
        "safety_factor": HOLDOUT_SAFETY_FACTOR,
        "acceptance_threshold": HOLDOUT_SAFETY_FACTOR * uncertainty,
        "reference_highest_valid_N": reference_N,
        "holdout_N": list(holdout_N),
        "weighted_error_contribution_J_m2": (
            None if weighted_row is None else weighted_row.get("weighted_error_contribution_J_m2")
        ),
    }


def _build_holdout_plan(
    *,
    audit_path: Path,
    output_root: Path,
    source_profile: str,
    target_profile: str,
    projection_reports: Sequence[Mapping[str, Any]],
    max_points: int = 32,
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

    reasons: dict[tuple[str, int, str, str], set[str]] = {}
    mandatory: set[tuple[str, int, str, str]] = set()
    for report in projection_reports:
        for row in report.get("decisions", []):
            if not isinstance(row, Mapping) or not bool(row.get("decision_changed")):
                continue
            identity = row.get("identity")
            if not isinstance(identity, list) or len(identity) != 4:
                continue
            key = (str(identity[0]), int(identity[1]), str(identity[2]), str(identity[3]))
            if key in entries:
                mandatory.add(key)
                reasons.setdefault(key, set()).add("projection_decision_changed")

    for key in weighted:
        if key in entries:
            reasons.setdefault(key, set()).add("largest_quadrature_weighted_uncertainty")

    selected: list[tuple[str, int, str, str]] = sorted(mandatory)
    ranked_weighted = sorted(
        (key for key in weighted if key in entries and key not in mandatory),
        key=lambda key: float(weighted[key].get("weighted_error_contribution_J_m2") or 0.0),
        reverse=True,
    )
    for key in ranked_weighted:
        if len(selected) >= max_points:
            break
        selected.append(key)

    strata: dict[tuple[str, int], list[tuple[float, tuple[str, int, str, str]]]] = {}
    for key, entry in entries.items():
        try:
            uncertainty = _local_uncertainty(entry["point_result"])
        except (KeyError, TypeError, ValueError):
            continue
        strata.setdefault((key[0], key[1]), []).append((uncertainty, key))
    for stratum in sorted(strata):
        if len(selected) >= max_points:
            break
        controls = sorted(strata[stratum], key=lambda value: value[0])
        for _, key in controls:
            if key not in selected:
                selected.append(key)
                reasons.setdefault(key, set()).add("easy_control_for_pairing_and_matsubara")
                break

    if not selected:
        raise ValueError("qualification holdout selection is empty")
    items = [
        _holdout_item(
            key,
            entries[key],
            reasons=sorted(reasons.get(key, {"stratified_control"})),
            weighted_row=weighted.get(key),
        )
        for key in selected
    ]
    source_hashes = {
        str(report["pairing"]): dict(report["source_artifact_sha256"])
        for report in projection_reports
    }
    payload = {
        "schema": HOLDOUT_PLAN_SCHEMA,
        "created_at_utc": _utc_now(),
        "profile": target_profile,
        "source_profile": source_profile,
        "candidate_logdet_rtol": LOGDET_RTOL,
        "safety_factor": HOLDOUT_SAFETY_FACTOR,
        "audit_report": str(Path(audit_path).resolve()),
        "audit_value_sha256": _digest(audit),
        "source_artifact_sha256": source_hashes,
        "target_cache_sha256": target_cache_sha,
        "selection_count": len(items),
        "max_primary_points": int(max_points),
        "items": items,
        "independence_contract": {
            "candidate_frozen_before_execution": True,
            "results_must_not_retune_candidate": True,
            "two_predeclared_N_levels": True,
            "every_point_must_pass": True,
        },
    }
    payload["plan_sha256"] = _digest(payload)
    return payload


def _prepare(args: argparse.Namespace) -> int:
    pairings = validate_pairings(args.pairings)
    if set(pairings) != {"spm", "dwave"}:
        raise ValueError("qualification preparation requires both spm and dwave")
    resources = select_runtime_resources(
        reserve_logical_cpus=int(args.reserve_cpus),
        worker_cap=int(args.worker_cap),
    )
    reports = []
    for pairing in pairings:
        target_run = Path(args.output_root) / case_name(pairing, 0, profile=args.profile)
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
    plan = _build_holdout_plan(
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
    print(f"plan_sha256: {plan['plan_sha256']}")
    print(f"selected CPUs: {resources.selected_cpus}")
    print("No microscopic holdout work was executed.")
    return 0


def _verify_bound_inputs(plan: Mapping[str, Any], output_root: Path) -> None:
    for pairing in ("spm", "dwave"):
        source_run = output_root / case_name(pairing, 0, profile=str(plan["source_profile"]))
        observed_source = _source_hashes(source_run)
        expected_source = plan["source_artifact_sha256"][pairing]
        if observed_source != expected_source:
            raise ValueError(f"source {pairing} changed after holdout planning")
        target_run = output_root / case_name(pairing, 0, profile=str(plan["profile"]))
        cache_path = target_run / "cache" / "certified_points.json"
        if _sha(cache_path) != plan["target_cache_sha256"][pairing]:
            raise ValueError(f"target {pairing} cache changed after holdout planning")


def _holdout(args: argparse.Namespace) -> int:
    plan = _read(Path(args.plan))
    if not isinstance(plan, Mapping) or plan.get("schema") != HOLDOUT_PLAN_SCHEMA:
        raise ValueError(f"holdout plan must use schema {HOLDOUT_PLAN_SCHEMA}")
    if str(args.confirm_plan_sha256) != str(plan.get("plan_sha256")):
        raise ValueError("holdout plan confirmation SHA-256 does not match")
    output_root = Path(args.output_root)
    _verify_bound_inputs(plan, output_root)
    grouped: dict[tuple[str, int, int, int, int], list[Mapping[str, Any]]] = {}
    for item in plan.get("items", []):
        if not isinstance(item, Mapping):
            raise ValueError("holdout plan contains a malformed item")
        holdout_N = item.get("holdout_N")
        if not isinstance(holdout_N, list) or len(holdout_N) != 2:
            raise ValueError("holdout item must contain two N levels")
        key = (
            str(item["pairing"]),
            int(item["n"]),
            int(item["candidate_audit_N"]),
            int(holdout_N[0]),
            int(holdout_N[1]),
        )
        grouped.setdefault(key, []).append(item)

    results: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="lno327-qualification-holdout-") as temporary:
        temp_root = Path(temporary)
        for group_index, (key, items) in enumerate(sorted(grouped.items())):
            pairing, n, candidate_N, holdout_1, holdout_2 = key
            target_run = output_root / case_name(pairing, 0, profile=str(plan["profile"]))
            full_config = _read_json_mapping(target_run / "config.json", label="target config")
            point_config = _point_config_from_run_config(full_config)
            run_config = replace(
                point_config,
                pairings=(pairing,),
                matsubara_indices=(n,),
                N_candidates=(candidate_N, holdout_1, holdout_2),
                required_consecutive_passes=2,
                logdet_rtol=0.0,
                logdet_atol=0.0,
            )
            labels = tuple(f"holdout_{group_index:04d}_{index:04d}" for index in range(len(items)))
            q_values = np.asarray([item["q_model"] for item in items], dtype=float)
            manifest = OuterQNodeManifest(
                labels=labels,
                q_model=q_values,
                grids={},
                labels_by_spec={},
            )
            output = temp_root / f"group_{group_index:04d}.json"
            started = perf_counter()
            certification = run_strict_transverse_certifier(run_config, manifest, output)
            wall_seconds = perf_counter() - started
            point_rows = certification.payload.get("point_results", [])
            by_label = {
                str(row.get("q_label")): row
                for row in point_rows
                if isinstance(row, Mapping)
            }
            group_reports.append(
                {
                    "group": list(key),
                    "point_count": len(items),
                    "wall_seconds": wall_seconds,
                    "execution_levels": certification.payload.get("execution_levels", []),
                    "stdout_tail": certification.stdout[-4000:],
                    "stderr_tail": certification.stderr[-4000:],
                }
            )
            for label, item in zip(labels, items, strict=True):
                row = by_label.get(label)
                if not isinstance(row, Mapping):
                    raise RuntimeError(f"holdout certifier omitted {label}")
                history_by_N = {
                    int(history["N"]): history
                    for history in row.get("history", [])
                    if isinstance(history, Mapping)
                }
                candidate_values = item["candidate_values_by_shift"]
                threshold = float(item["acceptance_threshold"])
                levels = []
                point_passed = True
                for N in (holdout_1, holdout_2):
                    history = history_by_N.get(N)
                    if not isinstance(history, Mapping):
                        raise RuntimeError(f"holdout history omitted N={N} for {label}")
                    shifts = history.get("shifts")
                    if not isinstance(shifts, Mapping) or set(shifts) != set(candidate_values):
                        raise RuntimeError(f"holdout shifts differ from candidate for {label}")
                    deltas = []
                    hard = True
                    for shift_label, state in shifts.items():
                        if not isinstance(state, Mapping):
                            hard = False
                            continue
                        hard = hard and bool(state.get("hard_physical_passed"))
                        value = float(state["two_plate_logdet"])
                        deltas.append(abs(value - float(candidate_values[shift_label])))
                    maximum_delta = max(deltas, default=math.inf)
                    passed = bool(hard and math.isfinite(maximum_delta) and maximum_delta <= threshold)
                    point_passed = point_passed and passed
                    levels.append(
                        {
                            "N": N,
                            "all_hard_physical_gates_passed": hard,
                            "maximum_shiftwise_absolute_delta": maximum_delta,
                            "acceptance_threshold": threshold,
                            "passed": passed,
                        }
                    )
                results.append(
                    {
                        "identity": list(item["identity"]),
                        "reasons": list(item["reasons"]),
                        "predicted_local_uncertainty": item["predicted_local_uncertainty"],
                        "safety_factor": item["safety_factor"],
                        "levels": levels,
                        "passed": point_passed,
                    }
                )

    _verify_bound_inputs(plan, output_root)
    report = {
        "schema": HOLDOUT_EXECUTION_SCHEMA,
        "created_at_utc": _utc_now(),
        "plan_path": str(Path(args.plan).resolve()),
        "plan_sha256": plan["plan_sha256"],
        "profile": plan["profile"],
        "result_count": len(results),
        "results": results,
        "group_reports": group_reports,
        "all_points_passed": bool(results) and all(bool(row["passed"]) for row in results),
        "source_v4_unchanged": True,
        "target_seed_caches_unchanged": True,
        "real_microscopic_work_executed": True,
        "candidate_retuning_forbidden": True,
    }
    report["execution_sha256"] = _digest(report)
    _write(Path(args.output), report)
    print(f"written: {Path(args.output).resolve()}")
    print(f"holdout points: {report['result_count']}")
    print(f"all_points_passed: {report['all_points_passed']}")
    print(f"execution_sha256: {report['execution_sha256']}")
    return 0 if report["all_points_passed"] else 2


def _assert_frozen_config(payload: Mapping[str, Any]) -> None:
    outer = payload["outer_tail_config"]
    joint = outer["joint_config"]
    point = joint["radial_config"]["point_config"]
    checks = {
        "logdet_rtol": math.isclose(float(point["logdet_rtol"]), LOGDET_RTOL, rel_tol=0.0, abs_tol=1e-15),
        "logdet_atol": math.isclose(float(point["logdet_atol"]), LOGDET_ATOL, rel_tol=0.0, abs_tol=1e-18),
        "required_consecutive_passes": int(point["required_consecutive_passes"]) == REQUIRED_CONSECUTIVE_PASSES,
        "N_candidates": tuple(point["N_candidates"]) == N_CANDIDATES,
        "radial_budget_fraction": math.isclose(float(joint["radial_budget_fraction"]), RADIAL_BUDGET_FRACTION),
        "angular_budget_fraction": math.isclose(float(joint["angular_budget_fraction"]), 1.0 - RADIAL_BUDGET_FRACTION),
        "outer_cutoffs": tuple(float(value) for value in outer["cutoff_u_values"]) == OUTER_CUTOFFS_U,
        "matsubara_cutoffs": tuple(int(value) for value in payload["matsubara_cutoff_values"]) == MATSUBARA_CUTOFFS,
        "total_rtol": math.isclose(float(payload["total_free_energy_rtol"]), TOTAL_RTOL),
        "total_atol": math.isclose(float(payload["total_free_energy_atol_J_m2"]), TOTAL_ATOL_J_M2),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"qualification config violates frozen policy: {failed}")


def _preflight(args: argparse.Namespace) -> int:
    _require_clean_git()
    holdout = _read(Path(args.holdout_report))
    if not isinstance(holdout, Mapping) or holdout.get("schema") != HOLDOUT_EXECUTION_SCHEMA:
        raise ValueError(f"holdout report must use schema {HOLDOUT_EXECUTION_SCHEMA}")
    payload_without_hash = dict(holdout)
    stored_execution_sha = payload_without_hash.pop("execution_sha256", None)
    if stored_execution_sha != _digest(payload_without_hash):
        raise ValueError("holdout execution self digest does not match")
    if holdout.get("all_points_passed") is not True:
        raise ValueError("qualification holdout did not pass")

    output_root = Path(args.output_root)
    named_configs: list[tuple[str, Mapping[str, Any]]] = []
    runs: dict[str, Any] = {}
    for pairing in ("spm", "dwave"):
        source_run = output_root / case_name(pairing, 0, profile=str(args.source_profile))
        target_run = output_root / case_name(pairing, 0, profile=str(args.profile))
        config_path = target_run / "config.json"
        cache_path = target_run / "cache" / "certified_points.json"
        projection_path = target_run / "cache" / "projection_report.json"
        config = _read_json_mapping(config_path, label="qualification config")
        projection = _read_json_mapping(projection_path, label="projection report")
        _assert_frozen_config(config)
        if projection.get("source_artifact_sha256") != _source_hashes(source_run):
            raise ValueError(f"source {pairing} changed after projection")
        if projection.get("target_cache_sha256") != _sha(cache_path):
            raise ValueError(f"target {pairing} cache changed before preflight")
        named_configs.append((pairing, config))
        runs[pairing] = {
            "source_run": str(source_run.resolve()),
            "target_run": str(target_run.resolve()),
            "source_artifact_sha256": dict(projection["source_artifact_sha256"]),
            "target_config_sha256": _sha(config_path),
            "target_cache_sha256": _sha(cache_path),
            "projection_sha256": projection["projection_sha256"],
        }
    parity = compare_policy_snapshots(named_configs)
    if parity.get("pairing_blind_scientific_policy") is not True:
        raise ValueError("SPM and d-wave qualification configs are not pairing blind")
    commit = _git_commit()
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "created_at_utc": _utc_now(),
        "status": "ready_to_run",
        "profile": str(args.profile),
        "source_profile": str(args.source_profile),
        "git_commit": commit,
        "frozen_policy": {
            "logdet_rtol": LOGDET_RTOL,
            "logdet_atol": LOGDET_ATOL,
            "required_consecutive_passes": REQUIRED_CONSECUTIVE_PASSES,
            "N_candidates": list(N_CANDIDATES),
            "radial_budget_fraction": RADIAL_BUDGET_FRACTION,
            "angular_budget_fraction": 1.0 - RADIAL_BUDGET_FRACTION,
            "outer_cutoffs_u": list(OUTER_CUTOFFS_U),
            "matsubara_cutoffs": list(MATSUBARA_CUTOFFS),
            "total_free_energy_rtol": TOTAL_RTOL,
            "total_free_energy_atol_J_m2": TOTAL_ATOL_J_M2,
        },
        "runs": runs,
        "holdout_report": str(Path(args.holdout_report).resolve()),
        "holdout_execution_sha256": stored_execution_sha,
        "pairing_blind_policy_audit": parity,
        "qualification_runner": "lno327.casimir.qualification.run_qualification_casimir",
        "source_v4_cache_immutable": True,
        "empty_restart_forbidden": True,
    }
    report["preflight_sha256"] = _digest(report)
    _write(Path(args.output), report)
    print(f"written: {Path(args.output).resolve()}")
    print("status: ready_to_run")
    print(f"git_commit: {commit}")
    print(f"preflight_sha256: {report['preflight_sha256']}")
    return 0


def _config_kwargs(payload: Mapping[str, Any], pairing: str) -> dict[str, Any]:
    outer = payload["outer_tail_config"]
    joint = outer["joint_config"]
    radial = joint["radial_config"]
    point = radial["point_config"]
    return {
        "pairings": (pairing,),
        "temperature_K": float(point["temperature_K"]),
        "separation_nm": float(point["separation_nm"]),
        "plate_angles_deg": tuple(float(value) for value in point["plate_angles_deg"]),
        "delta0_eV": float(point["delta0_eV"]),
        "eta_eV": float(point["eta_eV"]),
        "degeneracy": float(point["degeneracy"]),
        "N_candidates": tuple(int(value) for value in point["N_candidates"]),
        "required_consecutive_passes": int(point["required_consecutive_passes"]),
        "logdet_rtol": float(point["logdet_rtol"]),
        "logdet_atol": float(point["logdet_atol"]),
        "workers": int(point["workers"]),
        "parallel_mode": str(point["parallel_mode"]),
        "memory_budget_gb": float(point["memory_budget_gb"]),
        "max_context_workers": int(point["max_context_workers"]),
        "cutoff_u_values": tuple(float(value) for value in outer["cutoff_u_values"]),
        "outer_tail_start_u": float(outer["tail_start_u"]),
        "outer_tail_window_shells": int(outer["tail_window_shells"]),
        "outer_tail_ratio_max": float(outer["tail_ratio_max"]),
        "matsubara_cutoff_values": tuple(int(value) for value in payload["matsubara_cutoff_values"]),
        "matsubara_tail_start_n": int(payload["tail_start_n"]),
        "matsubara_tail_window_terms": int(payload["tail_window_terms"]),
        "matsubara_tail_ratio_max": float(payload["tail_ratio_max"]),
        "total_free_energy_rtol": float(payload["total_free_energy_rtol"]),
        "total_free_energy_atol_J_m2": float(payload["total_free_energy_atol_J_m2"]),
        "radial_budget_fraction": float(joint["radial_budget_fraction"]),
        "max_total_microscopic_q_nodes": int(outer["max_total_microscopic_q_nodes"]),
        "max_total_microscopic_point_entries": int(payload["max_total_microscopic_point_entries"]),
        "certifier_q_batch_size": int(payload["certifier_q_batch_size"]),
    }


def _run(args: argparse.Namespace) -> int:
    _require_clean_git()
    preflight = _read(Path(args.preflight))
    if not isinstance(preflight, Mapping) or preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError(f"preflight must use schema {PREFLIGHT_SCHEMA}")
    payload_without_hash = dict(preflight)
    stored_sha = payload_without_hash.pop("preflight_sha256", None)
    if stored_sha != _digest(payload_without_hash):
        raise ValueError("preflight self digest does not match")
    if str(args.confirm_preflight_sha256) != str(stored_sha):
        raise ValueError("preflight confirmation SHA-256 does not match")
    if preflight.get("status") != "ready_to_run":
        raise ValueError("qualification preflight is not ready_to_run")
    if preflight.get("git_commit") != _git_commit():
        raise ValueError("qualification checkout commit differs from preflight")
    output_root = Path(args.output_root)
    statuses = []
    for pairing in ("spm", "dwave"):
        run_record = preflight["runs"][pairing]
        source_run = Path(str(run_record["source_run"]))
        target_run = Path(str(run_record["target_run"]))
        if _source_hashes(source_run) != run_record["source_artifact_sha256"]:
            raise ValueError(f"source {pairing} changed after preflight")
        config_path = target_run / "config.json"
        cache_path = target_run / "cache" / "certified_points.json"
        if _sha(config_path) != run_record["target_config_sha256"]:
            raise ValueError(f"target {pairing} config changed after preflight")
        if _sha(cache_path) != run_record["target_cache_sha256"]:
            raise ValueError(f"target {pairing} seed cache changed after preflight")
        config = _read_json_mapping(config_path, label="qualification config")
        _assert_frozen_config(config)
        case = case_name(pairing, 0, profile=str(preflight["profile"]))
        result = execute_case(
            case=case,
            output_root=output_root,
            resume=True,
            runner=run_qualification_casimir,
            **_config_kwargs(config, pairing),
        )
        if _source_hashes(source_run) != run_record["source_artifact_sha256"]:
            raise RuntimeError(f"source {pairing} changed during qualification run")
        statuses.append(bool(result.matsubara_converged))
        print(
            f"qualification {pairing}: converged={result.matsubara_converged} "
            f"reason={result.termination_reason}",
            flush=True,
        )
    return 0 if all(statuses) else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.qualification",
        description="Prepare and run the frozen pairing-blind 0-degree qualification.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--pairings", nargs="+", default=("spm", "dwave"), choices=("spm", "dwave"))
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare.add_argument("--source-profile", default=SOURCE_PROFILE)
    prepare.add_argument("--profile", default=PROFILE)
    prepare.add_argument(
        "--audit-report",
        type=Path,
        default=Path("outputs/casimir/reports/convergence_audit.compact.json"),
    )
    prepare.add_argument(
        "--holdout-plan",
        type=Path,
        default=Path("outputs/casimir/catalog/0deg_qualification_v5_holdout_plan.json"),
    )
    prepare.add_argument("--max-holdout-points", type=int, default=32)
    prepare.add_argument("--reserve-cpus", type=int, default=6)
    prepare.add_argument("--worker-cap", type=int, default=26)
    prepare.add_argument("--memory-budget-gb", type=float, default=16.0)
    prepare.add_argument("--max-context-workers", type=int, default=1)
    prepare.add_argument("--parallel-mode", choices=("auto", "serial", "q", "context", "wave"), default="q")
    prepare.add_argument("--certifier-q-batch-size", type=int, default=512)

    holdout = commands.add_parser("holdout")
    holdout.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    holdout.add_argument("--plan", type=Path, required=True)
    holdout.add_argument("--confirm-plan-sha256", required=True)
    holdout.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/casimir/reports/0deg_qualification_v5_holdout.json"),
    )

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    preflight.add_argument("--source-profile", default=SOURCE_PROFILE)
    preflight.add_argument("--profile", default=PROFILE)
    preflight.add_argument("--holdout-report", type=Path, required=True)
    preflight.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/casimir/catalog/0deg_qualification_v5_preflight.json"),
    )

    run = commands.add_parser("run")
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--confirm-preflight-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "holdout":
            return _holdout(args)
        if args.command == "preflight":
            return _preflight(args)
        if args.command == "run":
            return _run(args)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"QUALIFICATION FAILED: {type(exc).__name__}: {exc}")
        return 2
    raise AssertionError(f"unhandled qualification command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
