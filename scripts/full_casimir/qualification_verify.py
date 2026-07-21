from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cache_migration import _read_json_mapping
from .data_management import _digest, _read, _sha, _write
from .policy_audit import compare_policy_snapshots
from .qualification import (
    PREFLIGHT_SCHEMA,
    _source_hashes,
)

FINAL_SCHEMA = "zero-degree-qualification-final-verification-v1"


def _pairing_result_passed(result: Mapping[str, Any], pairing: str) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if result.get("matsubara_converged") is not True:
        failures.append("matsubara_not_converged")
    if result.get("all_microscopic_nodes_certified") is not True:
        failures.append("microscopic_points_not_all_certified")
    if result.get("all_outer_tail_runs_converged") is not True:
        failures.append("outer_tail_runs_not_all_converged")
    if result.get("outer_tail_estimated") is not True:
        failures.append("outer_tail_not_certified")
    if result.get("matsubara_tail_estimated") is not True:
        failures.append("matsubara_tail_not_certified")
    pairings = result.get("pairing_results")
    payload = pairings.get(pairing) if isinstance(pairings, Mapping) else None
    if not isinstance(payload, Mapping):
        failures.append("pairing_result_missing")
        return False, failures
    if payload.get("finite_matsubara_budget_passed") is not True:
        failures.append("finite_matsubara_budget_failed")
    if payload.get("matsubara_tail_budget_passed") is not True:
        failures.append("matsubara_tail_budget_failed")
    if payload.get("total_free_energy_budget_passed") is not True:
        failures.append("total_free_energy_budget_failed")
    path = payload.get("outer_tail_certificate_path")
    if path not in {"geometric_numerical", "analytic_passive_vacuum"}:
        failures.append("valid_outer_tail_certificate_path_missing")
    if path == "analytic_passive_vacuum":
        contraction = payload.get("power_metric_contraction_premise")
        if not isinstance(contraction, Mapping) or contraction.get("all_points_certified") is not True:
            failures.append("analytic_tail_contraction_certificate_missing")
        analytic = payload.get("passive_vacuum_tail_certificate")
        if not isinstance(analytic, Mapping) or analytic.get("pairing_independent") is not True:
            failures.append("analytic_tail_certificate_missing")
    try:
        total_error = float(payload["estimated_total_error_J_m2"])
        tolerance = float(payload["total_free_energy_tolerance_J_m2"])
        if total_error > tolerance:
            failures.append("reported_total_error_exceeds_tolerance")
    except (KeyError, TypeError, ValueError, OverflowError):
        failures.append("total_error_ledger_missing")
    return not failures, failures


def verify_final(
    *,
    preflight_path: Path,
    confirm_preflight_sha256: str,
) -> dict[str, Any]:
    preflight = _read(Path(preflight_path))
    if not isinstance(preflight, Mapping) or preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError(f"preflight must use schema {PREFLIGHT_SCHEMA}")
    payload = dict(preflight)
    stored_sha = payload.pop("preflight_sha256", None)
    if stored_sha != _digest(payload):
        raise ValueError("preflight self digest does not match")
    if str(confirm_preflight_sha256) != str(stored_sha):
        raise ValueError("preflight confirmation SHA-256 does not match")
    if preflight.get("status") != "ready_to_run":
        raise ValueError("preflight is not ready_to_run")

    configs: list[tuple[str, Mapping[str, Any]]] = []
    run_results: dict[str, Any] = {}
    all_passed = True
    for pairing in ("spm", "dwave"):
        record = preflight["runs"][pairing]
        source_run = Path(str(record["source_run"]))
        target_run = Path(str(record["target_run"]))
        source_unchanged = _source_hashes(source_run) == record["source_artifact_sha256"]
        config = _read_json_mapping(target_run / "config.json", label="target config")
        manifest = _read_json_mapping(target_run / "manifest.json", label="target manifest")
        summary = _read_json_mapping(target_run / "summary.json", label="target summary")
        result = _read_json_mapping(target_run / "result.json", label="target result")
        configs.append((pairing, config))
        passed, failures = _pairing_result_passed(result, pairing)
        if manifest.get("status") != "completed":
            failures.append("manifest_not_completed")
        if manifest.get("git_commit") != preflight.get("git_commit"):
            failures.append("run_commit_differs_from_preflight")
        if summary.get("matsubara_converged") is not True:
            failures.append("summary_not_converged")
        if not source_unchanged:
            failures.append("source_v4_changed")
        passed = not failures
        all_passed = all_passed and passed
        run_results[pairing] = {
            "passed": passed,
            "failures": failures,
            "source_v4_unchanged": source_unchanged,
            "target_config_sha256": _sha(target_run / "config.json"),
            "target_manifest_sha256": _sha(target_run / "manifest.json"),
            "target_summary_sha256": _sha(target_run / "summary.json"),
            "target_result_sha256": _sha(target_run / "result.json"),
            "termination_reason": result.get("termination_reason"),
            "selected_matsubara_cutoff": result.get("selected_matsubara_cutoff"),
            "outer_tail_certificate_path": (
                result.get("pairing_results", {}).get(pairing, {}).get(
                    "outer_tail_certificate_path"
                )
                if isinstance(result.get("pairing_results"), Mapping)
                else None
            ),
        }

    parity = compare_policy_snapshots(configs)
    if parity.get("pairing_blind_scientific_policy") is not True:
        all_passed = False
    holdout_path = Path(str(preflight["holdout_report"]))
    holdout = _read(holdout_path)
    holdout_payload = dict(holdout) if isinstance(holdout, Mapping) else {}
    holdout_sha = holdout_payload.pop("execution_sha256", None)
    holdout_valid = bool(
        isinstance(holdout, Mapping)
        and holdout.get("all_points_passed") is True
        and holdout_sha == _digest(holdout_payload)
        and holdout_sha == preflight.get("holdout_execution_sha256")
    )
    all_passed = all_passed and holdout_valid
    report = {
        "schema": FINAL_SCHEMA,
        "status": "qualification_passed" if all_passed else "qualification_failed",
        "preflight_path": str(Path(preflight_path).resolve()),
        "preflight_sha256": stored_sha,
        "git_commit": preflight.get("git_commit"),
        "runs": run_results,
        "holdout_valid_and_passed": holdout_valid,
        "pairing_blind_policy_audit": parity,
        "same_tail_controller_structure": True,
        "same_error_allocation": parity.get("pairing_blind_scientific_policy") is True,
        "source_v4_caches_unchanged": all(
            bool(record["source_v4_unchanged"]) for record in run_results.values()
        ),
        "production_casimir_allowed": False,
        "interpretation": (
            "qualification_passed establishes the frozen numerical closure contract. "
            "The repository intentionally keeps production_casimir_allowed false until "
            "the broader scientific Casimir-use authorization is separately declared."
        ),
    }
    report["verification_sha256"] = _digest(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.full_casimir.qualification_verify",
        description="Verify both completed frozen 0-degree qualification runs.",
    )
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--confirm-preflight-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/casimir/reports/0deg_qualification_v5_final.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify_final(
            preflight_path=Path(args.preflight),
            confirm_preflight_sha256=str(args.confirm_preflight_sha256),
        )
        _write(Path(args.output), report)
        print(f"written: {Path(args.output).resolve()}")
        print(f"status: {report['status']}")
        print(f"verification_sha256: {report['verification_sha256']}")
        return 0 if report["status"] == "qualification_passed" else 2
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"QUALIFICATION VERIFY FAILED: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
