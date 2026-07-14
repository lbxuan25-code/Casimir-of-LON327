"""Formal same-head gate for the expensive arbitrary-q qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from lno327.response.arbitrary_q_formal_policy import (
    FORMAL_POLICY_ID,
    config_fingerprint,
    validate_numerical_formal_config,
    validate_performance_formal_config,
    validate_performance_manifest_compatibility,
)
from validation.commands.matsubara import (
    arbitrary_q_periodic_bz_qualification as qualification,
)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_manifest(
    path: Path,
    *,
    git_head: str,
    qualification_config: dict[str, Any],
) -> dict[str, Any]:
    if git_head == "unknown":
        raise SystemExit("qualification requires a resolvable git HEAD")
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SystemExit(f"performance manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"performance manifest is unreadable: {exc}") from exc

    if payload.get("schema") != "arbitrary-q-performance-preflight-v2":
        raise SystemExit("performance manifest has the wrong schema")
    if payload.get("git_head") != git_head:
        raise SystemExit(
            "performance manifest git_head does not match qualification HEAD: "
            f"manifest={payload.get('git_head')!r}, current={git_head!r}"
        )
    if payload.get("formal_policy_id") != FORMAL_POLICY_ID:
        raise SystemExit("performance manifest has the wrong formal policy id")
    if payload.get("formal_policy_passed") is not True:
        raise SystemExit("performance manifest did not pass the formal policy")
    if payload.get("arbitrary_q_performance_contract") != "formal_preflight_passed":
        raise SystemExit("performance manifest did not establish formal_preflight_passed")
    if payload.get("passed") is not True:
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
        "git_head": payload["git_head"],
        "created_at_utc": payload.get("created_at_utc"),
        "hardware_fingerprint": payload["hardware"]["hardware_fingerprint"],
        "exact_command": payload["exact_command"],
        "formal_policy_id": payload["formal_policy_id"],
        "config_fingerprint": payload["config_fingerprint"],
        "contract": payload["arbitrary_q_performance_contract"],
        "execution_strategy": manifest_config["execution_strategy"],
        "thread_policy_id": manifest_config["thread_policy_id"],
        "workers": manifest_config["workers"],
        "canonical_block_size": manifest_config["canonical_block_size"],
        "runtime_chunk_sizes": manifest_config["runtime_chunk_sizes"],
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
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"qualification output cannot record its formal gate: {exc}"
        ) from exc
    if payload.get("schema") != "arbitrary-q-periodic-bz-diagnostic-result-v2":
        raise RuntimeError("qualification core wrote an unexpected schema")
    if payload.get("passed") is not True:
        raise RuntimeError("qualification core did not pass")
    if payload.get("arbitrary_q_microscopic_contract") != "diagnostic_result_passed":
        raise RuntimeError("qualification core did not produce diagnostic_result_passed")
    if payload.get("formal_policy_id") != FORMAL_POLICY_ID:
        raise RuntimeError("qualification output has the wrong formal policy id")
    if payload.get("formal_policy_passed") is not True:
        raise RuntimeError("qualification output did not pass numerical formal policy")
    if payload.get("config_fingerprint") != config_fingerprint(numerical_config):
        raise RuntimeError("qualification output config fingerprint is invalid")

    payload["blocking_performance_manifest"] = performance_record
    payload["authorization_source"] = (
        "public_same_head_formal_gate_after_performance_and_numerical_pass"
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
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    values = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
    )
    parser.add_argument(
        "--performance-manifest",
        type=Path,
        required=True,
        help="same-git-head formal arbitrary-q performance preflight JSON",
    )
    gate_args, remaining = parser.parse_known_args(values)
    parsed = qualification._args(remaining)
    numerical_config = qualification._formal_config(parsed)
    numerical_policy = validate_numerical_formal_config(numerical_config)
    if not numerical_policy.passed:
        raise SystemExit(
            "qualification configuration is looser than the frozen formal policy: "
            + "; ".join(numerical_policy.violations)
        )
    git_head = _git_head()
    record = _load_manifest(
        gate_args.performance_manifest,
        git_head=git_head,
        qualification_config=numerical_config,
    )
    output = _output_path(remaining)
    qualification.main(remaining)
    _authorize_output(
        output,
        performance_record=record,
        numerical_config=numerical_config,
    )


if __name__ == "__main__":
    main()
