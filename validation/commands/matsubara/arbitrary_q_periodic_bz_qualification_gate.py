"""Same-head performance gate for the expensive arbitrary-q qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from validation.commands.matsubara import arbitrary_q_periodic_bz_qualification as qualification


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_manifest(path: Path, *, git_head: str) -> dict[str, Any]:
    if git_head == "unknown":
        raise SystemExit("qualification requires a resolvable git HEAD")
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise SystemExit(f"performance manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"performance manifest is unreadable: {exc}") from exc
    if payload.get("schema") != "arbitrary-q-performance-preflight-v1":
        raise SystemExit("performance manifest has the wrong schema")
    if payload.get("git_head") != git_head:
        raise SystemExit(
            "performance manifest git_head does not match qualification HEAD: "
            f"manifest={payload.get('git_head')!r}, current={git_head!r}"
        )
    if payload.get("arbitrary_q_performance_contract") != "preflight_passed":
        raise SystemExit("performance manifest did not establish preflight_passed")
    if payload.get("passed") is not True:
        raise SystemExit("performance manifest is not passed")
    if payload.get("production_reference_established") is not False:
        raise SystemExit("performance manifest has an invalid readiness state")
    if payload.get("valid_for_casimir_input") is not False:
        raise SystemExit("performance manifest has an invalid Casimir readiness state")
    return {
        "path": str(manifest_path),
        "schema": payload["schema"],
        "git_head": payload["git_head"],
        "created_at_utc": payload.get("created_at_utc"),
        "platform": payload.get("platform"),
        "contract": payload["arbitrary_q_performance_contract"],
        "passed": True,
    }


def _output_path(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, default=qualification.DEFAULT_OUTPUT)
    args, _unknown = parser.parse_known_args(list(argv))
    return Path(args.output)


def _inject_gate_record(path: Path, record: dict[str, Any]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"qualification output cannot record its performance gate: {exc}"
        ) from exc
    payload["blocking_performance_manifest"] = record
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
        help="same-git-head passed arbitrary-q performance preflight JSON",
    )
    gate_args, remaining = parser.parse_known_args(values)
    git_head = _git_head()
    record = _load_manifest(gate_args.performance_manifest, git_head=git_head)
    output = _output_path(remaining)
    qualification.main(remaining)
    _inject_gate_record(output, record)


if __name__ == "__main__":
    main()
