from __future__ import annotations

import json

from scripts.full_casimir import __main__ as unified_cli
from scripts.full_casimir import scan
from scripts.full_casimir.config import (
    angle_token,
    case_name,
    inclusive_float_grid,
    physical_case_name,
)


def test_inclusive_float_grid_is_decimal_exact() -> None:
    assert inclusive_float_grid(0.0, 1.0, 0.25) == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_default_legacy_case_name_remains_backward_compatible() -> None:
    assert case_name("spm", 0) == (
        "spm_T10K_d20nm_theta_p000deg_runtime_budget_v3"
    )


def test_formal_case_name_has_no_human_version_suffix() -> None:
    assert angle_token(-2.5) == "m002p5"
    assert physical_case_name(
        "dwave",
        2.5,
        temperature_K=12.5,
        separation_nm=17.25,
    ) == "dwave_T12p5K_d17p25nm_theta_p002p5deg"


def _code_identity() -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "tracked_worktree_clean": True,
    }


def test_plan_freezes_campaign_policy_and_case_identities(capsys, monkeypatch) -> None:
    monkeypatch.setattr(scan, "git_code_identity", _code_identity)
    status = scan.main(
        [
            "plan",
            "--pairings",
            "spm",
            "--distances-nm",
            "10",
            "20",
            "--angles-deg",
            "0",
            "2.5",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "cases: 4" in output
    assert "campaign_id: campaign-" in output
    assert "spm_T10K_d10nm_theta_p000deg" in output
    assert "spm_T10K_d20nm_theta_p002p5deg" in output
    assert "runtime_budget_v3" not in output


def test_plan_file_self_digest_and_execution_settings_are_separate(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(scan, "git_code_identity", _code_identity)
    plan_path = tmp_path / "plan.json"
    status = scan.main(
        [
            "plan",
            "--pairings",
            "dwave",
            "--distances-nm",
            "20",
            "--angles-deg",
            "0",
            "--plan-output",
            str(plan_path),
        ]
    )
    assert status == 0
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "full-casimir-production-plan-v1"
    assert payload["plan_sha256"]
    assert payload["scientific_policy_sha256"]
    serialized = json.dumps(payload, sort_keys=True)
    assert "worker_cap" not in serialized
    assert "parallel_mode" not in serialized
    assert "memory_budget_gb" not in serialized


def test_plan_accepts_inclusive_range_syntax(capsys, monkeypatch) -> None:
    monkeypatch.setattr(scan, "git_code_identity", _code_identity)
    status = scan.main(
        [
            "plan",
            "--pairings",
            "dwave",
            "--distance-min-nm",
            "10",
            "--distance-max-nm",
            "20",
            "--distance-step-nm",
            "5",
            "--angle-min-deg",
            "0",
            "--angle-max-deg",
            "4",
            "--angle-step-deg",
            "2",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "cases: 9" in output


def test_explicit_and_range_axis_syntax_cannot_be_mixed(capsys, monkeypatch) -> None:
    monkeypatch.setattr(scan, "git_code_identity", _code_identity)
    status = scan.main(
        [
            "plan",
            "--angles-deg",
            "0",
            "--angle-min-deg",
            "0",
            "--angle-max-deg",
            "2",
            "--angle-step-deg",
            "1",
        ]
    )

    assert status == 2
    assert "cannot be combined" in capsys.readouterr().out


def test_run_requires_plan_confirmation_and_explicit_mode() -> None:
    parser = scan._parser()
    try:
        parser.parse_args(["run", "--plan", "plan.json", "--confirm-plan-sha256", "x"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - argparse contract
        raise AssertionError("run accepted no fresh/resume mode")


def test_unified_entry_routes_primary_commands(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_loader(module_name: str):
        seen["module"] = module_name

        def fake_main(argv):
            seen["argv"] = list(argv)
            return 7

        return fake_main

    monkeypatch.setattr(unified_cli, "_module_main", fake_loader)

    assert unified_cli.main(["plan", "--angles-deg", "0", "45"]) == 7
    assert seen == {
        "module": "scripts.full_casimir.scan",
        "argv": ["plan", "--angles-deg", "0", "45"],
    }
