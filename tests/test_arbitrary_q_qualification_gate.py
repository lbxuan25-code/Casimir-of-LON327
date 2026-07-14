from __future__ import annotations

import json

import pytest

from validation.commands.matsubara.arbitrary_q_periodic_bz_qualification_gate import (
    _load_manifest,
)


def _manifest(head: str, *, passed: bool = True) -> dict[str, object]:
    return {
        "schema": "arbitrary-q-performance-preflight-v1",
        "git_head": head,
        "created_at_utc": "2026-07-14T00:00:00+00:00",
        "platform": "test",
        "arbitrary_q_performance_contract": (
            "preflight_passed" if passed else "preflight_failed"
        ),
        "diagnostic_only": True,
        "production_reference_established": False,
        "valid_for_casimir_input": False,
        "passed": passed,
    }


def test_qualification_gate_accepts_only_same_head_passed_manifest(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_manifest("abc123")), encoding="utf-8")
    record = _load_manifest(path, git_head="abc123")
    assert record["passed"] is True
    assert record["git_head"] == "abc123"
    assert record["contract"] == "preflight_passed"


def test_qualification_gate_rejects_stale_or_failed_manifest(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_manifest("old-head")), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match"):
        _load_manifest(path, git_head="new-head")

    path.write_text(json.dumps(_manifest("new-head", passed=False)), encoding="utf-8")
    with pytest.raises(SystemExit, match="preflight_passed"):
        _load_manifest(path, git_head="new-head")


def test_qualification_gate_rejects_unknown_current_head(tmp_path) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(_manifest("abc123")), encoding="utf-8")
    with pytest.raises(SystemExit, match="resolvable git HEAD"):
        _load_manifest(path, git_head="unknown")
