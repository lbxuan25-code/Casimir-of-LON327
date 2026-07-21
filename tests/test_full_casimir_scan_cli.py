from __future__ import annotations

from scripts.full_casimir import __main__ as unified_cli
from scripts.full_casimir import scan
from scripts.full_casimir.config import (
    angle_token,
    case_name,
    inclusive_float_grid,
)


def test_inclusive_float_grid_is_decimal_exact() -> None:
    assert inclusive_float_grid(0.0, 1.0, 0.25) == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_default_case_name_remains_backward_compatible() -> None:
    assert case_name("spm", 0) == (
        "spm_T10K_d20nm_theta_p000deg_runtime_budget_v3"
    )


def test_case_name_encodes_nondefault_physical_identity() -> None:
    assert angle_token(-2.5) == "m002p5"
    assert case_name(
        "dwave",
        2.5,
        temperature_K=12.5,
        separation_nm=17.25,
        profile="policy-test",
    ) == "dwave_T12p5K_d17p25nm_theta_p002p5deg_policy-test"


def test_plan_accepts_single_or_multiple_distances_and_angles(capsys) -> None:
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
            "--profile",
            "policy-test",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "cases: 4" in output
    assert "spm_T10K_d10nm_theta_p000deg_policy-test" in output
    assert "spm_T10K_d20nm_theta_p002p5deg_policy-test" in output


def test_plan_accepts_inclusive_range_syntax(capsys) -> None:
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


def test_explicit_and_range_axis_syntax_cannot_be_mixed(capsys) -> None:
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
