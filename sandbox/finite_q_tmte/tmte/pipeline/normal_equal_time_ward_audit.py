"""Diagnostic-only normal finite-q equal-time Ward audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io.writers import write_json

SCHEMA_VERSION = "finite_q_tmte_normal_equal_time_ward_audit_v1"


def run_normal_equal_time_ward_audit(**kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "diagnostic_run_completed": False,
            "diagnostic_only_not_a_fix": True,
            "accepted_convention": False,
            "valid_for_casimir_input": False,
            "reason": "normal_equal_time_ward_audit_skeleton",
        },
        "debug_parameters": dict(kwargs),
        "valid_for_casimir_input": False,
    }


def run_and_write_normal_equal_time_ward_audit(output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    payload = run_normal_equal_time_ward_audit(**kwargs)
    write_json(Path(output_dir) / "normal_equal_time_ward_audit.json", payload)
    return payload
