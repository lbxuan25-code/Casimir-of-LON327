from __future__ import annotations

from pathlib import Path
import json
import math
import subprocess
import sys

import pytest

from lno327.casimir.certified_point_provider import certified_point_policy_fingerprint
from lno327.casimir.fixed_chain import FixedCasimirConfig
from scripts.full_casimir import workflow
from scripts.full_casimir.cache_migration import CACHE_SCHEMA, migrate_cache
from scripts.full_casimir.cleanup_legacy_root import cleanup_legacy_root_scripts
from scripts.full_casimir.config import (
    REPO_ROOT,
    angle_token,
    case_name,
    inclusive_integer_grid,
    select_runtime_resources,
)
from scripts.full_casimir.postprocess import (
    five_point_torque,
    five_point_torque_error_bound,
)


def test_angle_grid_and_case_names_are_deterministic() -> None:
    assert inclusive_integer_grid(-4, 94, 2)[0] == -4
    assert inclusive_integer_grid(-4, 94, 2)[-1] == 94
    assert len(inclusive_integer_grid(-4, 94, 2)) == 50
    assert angle_token(-4) == "m004"
    assert angle_token(0) == "p000"
    assert angle_token(94) == "p094"
    assert case_name("spm", 0) == (
        "spm_T10K_d20nm_theta_p000deg_runtime_budget_v3"
    )


def test_cpu_selection_reserves_requested_logical_capacity() -> None:
    resources = select_runtime_resources(
        available_cpus=tuple(range(32)),
        reserve_logical_cpus=6,
        worker_cap=26,
    )
    assert resources.workers == 26
    assert len(resources.reserved_cpus) == 6
    assert set(resources.selected_cpus).isdisjoint(resources.reserved_cpus)
    assert set(resources.selected_cpus) | set(resources.reserved_cpus) == set(range(32))


def test_five_point_torque_is_exact_for_quartic() -> None:
    step_deg = 2
    angle_deg = 20
    angles = range(angle_deg - 4, angle_deg + 5, 2)
    energies = {angle: math.radians(angle) ** 4 for angle in angles}
    expected = -4.0 * math.radians(angle_deg) ** 3
    assert five_point_torque(
        energies,
        angle_deg=angle_deg,
        step_deg=step_deg,
    ) == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_torque_error_bound_uses_absolute_stencil_coefficients() -> None:
    errors = {angle: 1.0 for angle in (16, 18, 22, 24)}
    expected = 18.0 / (12.0 * math.radians(2))
    assert five_point_torque_error_bound(
        errors,
        angle_deg=20,
        step_deg=2,
    ) == pytest.approx(expected)


def test_cleanup_removes_only_explicit_legacy_names(tmp_path: Path) -> None:
    legacy = tmp_path / "run_full_casimir_N896_scan.sh"
    keep = tmp_path / "important.py"
    legacy.write_text("legacy\n", encoding="utf-8")
    keep.write_text("keep\n", encoding="utf-8")

    removed = cleanup_legacy_root_scripts(root=tmp_path)

    assert removed == [legacy]
    assert not legacy.exists()
    assert keep.exists()


def test_custom_pilot_profile_is_used_for_cache_migration(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(workflow, "cleanup_legacy_root_scripts", lambda: [])
    monkeypatch.setattr(workflow, "_resources", lambda args: object())
    monkeypatch.setattr(workflow, "validate_pairings", lambda values: tuple(values))
    monkeypatch.setattr(workflow, "_energy_options", lambda args: object())

    def fake_migrate(args, pairings, resources, options, *, target_profile):
        seen["target_profile"] = target_profile

    monkeypatch.setattr(workflow, "_migrate", fake_migrate)
    monkeypatch.setattr(workflow, "run_energy_cases", lambda **kwargs: 0)

    assert workflow.main(["pilots", "--profile", "custom_pilot_profile"]) == 0
    assert seen["target_profile"] == "custom_pilot_profile"


def test_existing_target_cache_must_match_target_policy(tmp_path: Path) -> None:
    target_config = FixedCasimirConfig(pairings=("spm",), plate_angles_deg=(0.0, 0.0))
    target_run = tmp_path / "target"
    target_cache = target_run / "cache" / "certified_points.json"
    target_cache.parent.mkdir(parents=True)
    target_cache.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "policy_fingerprint": "stale-policy",
                "frequency_extendable": True,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint"):
        migrate_cache(
            pairing="spm",
            source_run_dir=tmp_path / "source",
            target_run_dir=target_run,
            target_point_config=target_config,
        )

    assert certified_point_policy_fingerprint(
        target_config,
        frequency_extendable=True,
    ) != "stale-policy"


def test_background_runner_persists_child_exit_code(tmp_path: Path) -> None:
    runner = REPO_ROOT / "scripts" / "full_casimir" / "background_runner.sh"
    exit_file = tmp_path / "exit_code"

    completed = subprocess.run(
        [
            "bash",
            str(runner),
            str(exit_file),
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 7
    assert exit_file.read_text(encoding="utf-8").strip() == "7"
