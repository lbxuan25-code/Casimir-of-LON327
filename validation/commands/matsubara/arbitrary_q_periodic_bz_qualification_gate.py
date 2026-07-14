"""Formal clean-source gate for the expensive arbitrary-q qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lno327.response.arbitrary_q_formal_policy import (
    FORMAL_POLICY_ID,
    OUTER_Q_BATCH_WORKLOAD_ID,
    QUALIFICATION_AUDIT_WORKLOAD_ID,
    QUALIFICATION_PRIMARY_WORKLOAD_ID,
    config_fingerprint,
    validate_numerical_formal_config,
    validate_performance_formal_config,
    validate_performance_manifest_compatibility,
)
from validation.commands.matsubara import (
    arbitrary_q_periodic_bz_qualification as qualification,
)
from validation.lib.source_tree_provenance import (
    provenance_compatibility,
    source_tree_provenance,
)


def _provenance_mapping(payload: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: payload.get(key)
        for key in (
            "git_head",
            "git_tree_sha",
            "worktree_clean",
            "source_tree_fingerprint",
            "tracked_index_fingerprint",
        )
    }


def _workload_records(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for pairing in payload.get("pairings", ()):  # type: ignore[assignment]
        if not isinstance(pairing, Mapping):
            continue
        for record in pairing.get("records", ()):  # type: ignore[assignment]
            if not isinstance(record, Mapping):
                continue
            for workload in record.get("workloads", ()):  # type: ignore[assignment]
                if not isinstance(workload, Mapping):
                    continue
                identifier = str(workload.get("workload_id", ""))
                result.setdefault(identifier, []).append(dict(workload))
    return result


def _load_manifest(
    path: Path,
    *,
    current_provenance: dict[str, object],
    qualification_config: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SystemExit(f"performance manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"performance manifest is unreadable: {exc}") from exc

    if payload.get("schema") != "arbitrary-q-performance-preflight-v3":
        raise SystemExit("performance manifest has the wrong schema")
    provenance_errors = provenance_compatibility(
        _provenance_mapping(payload), current_provenance
    )
    if provenance_errors:
        raise SystemExit(
            "performance manifest source tree differs from qualification: "
            + "; ".join(provenance_errors)
        )
    if payload.get("formal_policy_id") != FORMAL_POLICY_ID:
        raise SystemExit("performance manifest has the wrong formal policy id")
    if payload.get("formal_policy_passed") is not True:
        raise SystemExit("performance manifest did not pass the formal policy")
    if payload.get("diagnostic_nonformal_requested") is True:
        raise SystemExit("nonformal performance evidence cannot authorize qualification")
    if payload.get("arbitrary_q_performance_contract") != "formal_preflight_passed":
        raise SystemExit("performance manifest did not establish formal_preflight_passed")
    if payload.get("passed") is not True or payload.get("metric_passed") is not True:
        raise SystemExit("performance manifest is not passed")
    if payload.get("actual_threadpool_passed") is not True:
        raise SystemExit("performance manifest did not verify actual BLAS thread counts")
    if not payload.get("hardware", {}).get("hardware_fingerprint"):
        raise SystemExit("performance manifest lacks a hardware fingerprint")
    if not payload.get("exact_command"):
        raise SystemExit("performance manifest lacks the exact command")
    if payload.get("production_reference_established") is not False:
        raise SystemExit("performance manifest has an invalid readiness state")
    if payload.get("valid_for_casimir_input") is not False:
        raise SystemExit("performance manifest has an invalid Casimir readiness state")

    manifest_config = payload.get("config")
    if not isinstance(manifest_config, dict):
        raise SystemExit("performance manifest lacks a config mapping")
    performance_policy = validate_performance_formal_config(manifest_config)
    if not performance_policy.passed:
        raise SystemExit(
            "performance manifest config violates the frozen policy: "
            + "; ".join(performance_policy.violations)
        )
    if payload.get("config_fingerprint") != config_fingerprint(manifest_config):
        raise SystemExit("performance manifest config fingerprint is invalid")

    workloads = _workload_records(payload)
    required = {
        OUTER_Q_BATCH_WORKLOAD_ID,
        QUALIFICATION_PRIMARY_WORKLOAD_ID,
        QUALIFICATION_AUDIT_WORKLOAD_ID,
    }
    missing = required.difference(workloads)
    if missing:
        raise SystemExit(
            "performance manifest lacks required workload evidence: "
            + ", ".join(sorted(missing))
        )
    for identifier in required:
        rows = workloads[identifier]
        if not rows or not all(row.get("passed") is True for row in rows):
            raise SystemExit(f"performance workload {identifier!r} did not pass")
        if any(
            int(row.get("parallel_metadata", {}).get("process_workers", -1)) > 1
            and float(
                row.get("parallel_metadata", {}).get("pool_shutdown_seconds", 0.0)
            ) <= 0.0
            for row in rows
        ):
            raise SystemExit(
                f"performance workload {identifier!r} did not measure pool shutdown"
            )

    compatibility = validate_performance_manifest_compatibility(
        manifest_config=manifest_config,
        qualification_config=qualification_config,
    )
    if compatibility:
        raise SystemExit(
            "performance manifest is incompatible with qualification: "
            + "; ".join(compatibility)
        )
    return {
        "path": str(manifest_path),
        "schema": payload["schema"],
        **_provenance_mapping(payload),
        "created_at_utc": payload.get("created_at_utc"),
        "hardware_fingerprint": payload["hardware"]["hardware_fingerprint"],
        "exact_command": payload["exact_command"],
        "formal_policy_id": payload["formal_policy_id"],
        "config_fingerprint": payload["config_fingerprint"],
        "contract": payload["arbitrary_q_performance_contract"],
        "execution_strategy": manifest_config["execution_strategy"],
        "thread_policy_id": manifest_config["thread_policy_id"],
        "runtime_chunk_sizes": manifest_config["runtime_chunk_sizes"],
        "workload_evidence": {
            identifier: [
                {
                    "pairing_record_count": len(workloads[identifier]),
                    "task_count": row.get("task_count"),
                    "workers": row.get("workers"),
                    "speedup": row.get("speedup"),
                    "pool_overhead_fraction": row.get("pool_overhead_fraction"),
                    "passed": row.get("passed"),
                }
                for row in workloads[identifier]
            ]
            for identifier in sorted(required)
        },
        "passed": True,
    }


def _output_path(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, default=qualification.DEFAULT_OUTPUT)
    args, _unknown = parser.parse_known_args(list(argv))
    return Path(args.output)


def _authorize_output(
    path: Path,
    *,
    performance_record: dict[str, Any],
    numerical_config: dict[str, Any],
    current_provenance: dict[str, object],
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"qualification output cannot record its formal gate: {exc}"
        ) from exc
    if payload.get("schema") != "arbitrary-q-periodic-bz-diagnostic-result-v3":
        raise RuntimeError("qualification core wrote an unexpected schema")
    if payload.get("passed") is not True:
        raise RuntimeError("qualification core did not pass")
    if payload.get("formal_evidence_eligible") is not True:
        raise RuntimeError("qualification core was not formal-evidence eligible")
    if payload.get("actual_threadpool_passed") is not True:
        raise RuntimeError("qualification core did not verify actual threadpools")
    if payload.get("diagnostic_nonformal_requested") is True:
        raise RuntimeError("nonformal qualification cannot be authorized")
    if payload.get("arbitrary_q_microscopic_contract") != "diagnostic_result_passed":
        raise RuntimeError("qualification core did not produce diagnostic_result_passed")
    if payload.get("formal_policy_id") != FORMAL_POLICY_ID:
        raise RuntimeError("qualification output has the wrong formal policy id")
    if payload.get("formal_policy_passed") is not True:
        raise RuntimeError("qualification output did not pass numerical formal policy")
    if payload.get("config_fingerprint") != config_fingerprint(numerical_config):
        raise RuntimeError("qualification output config fingerprint is invalid")
    provenance_errors = provenance_compatibility(
        _provenance_mapping(payload), current_provenance
    )
    if provenance_errors:
        raise RuntimeError(
            "qualification output source provenance changed during execution: "
            + "; ".join(provenance_errors)
        )
    performance_errors = provenance_compatibility(
        performance_record, current_provenance
    )
    if performance_errors:
        raise RuntimeError(
            "performance and numerical source provenance differ: "
            + "; ".join(performance_errors)
        )
    coverage = payload.get("numerical_q_coverage", {})
    if coverage.get("principal_supported_domain_is_not_claimed_as_qualified") is not True:
        raise RuntimeError("qualification output overclaims its q-domain coverage")
    if coverage.get("qualified_outer_q_envelope_established") is not False:
        raise RuntimeError("microscopic qualification must not invent an outer q envelope")

    payload["blocking_performance_manifest"] = performance_record
    payload["formal_gate_source_provenance"] = current_provenance
    payload["authorization_source"] = (
        "public_clean_same_source_formal_gate_after_performance_and_numerical_pass"
    )
    payload["arbitrary_q_microscopic_contract"] = (
        "qualified_for_diagnostic_outer_integration"
    )
    payload["formal_authorization_passed"] = True
    payload["diagnostic_only"] = True
    payload["production_reference_established"] = False
    payload["valid_for_casimir_input"] = False
    temporary = path.with_suffix(path.suffix + ".gate.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    values = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--performance-manifest",
        type=Path,
        required=True,
        help="clean same-source formal arbitrary-q performance preflight JSON",
    )
    gate_args, remaining = parser.parse_known_args(values)
    parsed = qualification._args(remaining)
    if parsed.diagnostic_nonformal:
        raise SystemExit("public formal gate rejects --diagnostic-nonformal")
    numerical_config = qualification._formal_config(parsed)
    numerical_policy = validate_numerical_formal_config(numerical_config)
    if not numerical_policy.passed:
        raise SystemExit(
            "qualification configuration is looser than the frozen formal policy: "
            + "; ".join(numerical_policy.violations)
        )
    provenance = source_tree_provenance()
    provenance.require_clean()
    current = provenance.as_dict()
    record = _load_manifest(
        gate_args.performance_manifest,
        current_provenance=current,
        qualification_config=numerical_config,
    )
    output = _output_path(remaining)
    qualification.main(remaining)
    post = source_tree_provenance()
    post.require_clean()
    post_errors = provenance_compatibility(current, post)
    if post_errors:
        raise SystemExit(
            "source tree changed during qualification: " + "; ".join(post_errors)
        )
    _authorize_output(
        output,
        performance_record=record,
        numerical_config=numerical_config,
        current_provenance=current,
    )


if __name__ == "__main__":
    main()
