from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import subprocess
import sys

import pytest

from lno327.casimir.certified_point_provider import (
    certified_point_policy_fingerprint,
    certified_point_policy_payload,
)
from lno327.casimir.fixed_chain import FixedCasimirConfig
from scripts.full_casimir import workflow
from scripts.full_casimir.cache_migration import (
    CACHE_SCHEMA,
    LEGACY_SCHEDULING_FIELDS,
    migrate_cache,
)
from scripts.full_casimir.cleanup_legacy_root import cleanup_legacy_root_scripts
from scripts.full_casimir.config import (
    REPO_ROOT,
    angle_token,
    case_name,
    inclusive_integer_grid,
    select_runtime_resources,
)
from scripts.full_casimir.energy import _case_state
from scripts.full_casimir.postprocess import (
    five_point_torque,
    five_point_torque_error_bound,
    postprocess_torque,
)


def _write_completed_artifacts(run: Path, *, config: dict) -> None:
    run.mkdir(parents=True, exist_ok=True)
    reason = "outer_and_matsubara_cutoff_tail_tolerances_met"
    common = {
        "selected_matsubara_cutoff": 1,
        "production_casimir_allowed": False,
        "provider_statistics": {},
    }
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "full-casimir-run-manifest",
                "case": run.name,
                "status": "completed",
                "termination_reason": reason,
            }
        ),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "schema": "full-casimir-run-summary",
                "case": run.name,
                "status": "adaptive_tail_bounded",
                "matsubara_converged": True,
                "termination_reason": reason,
                "pairings": {},
                **common,
            }
        ),
        encoding="utf-8",
    )
    (run / "result.json").write_text(
        json.dumps(
            {
                "schema": "adaptive-matsubara-casimir-result-v1",
                "status": "adaptive_tail_bounded",
                "matsubara_converged": True,
                "termination_reason": reason,
                "pairing_results": {},
                "config": config,
                **common,
            }
        ),
        encoding="utf-8",
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


def test_torque_metadata_does_not_claim_truncation_error_is_bounded(
    tmp_path: Path,
) -> None:
    energy_csv, torque_csv, metadata_json, complete = postprocess_torque(
        run_root=tmp_path / "runs",
        output_root=tmp_path / "postprocessed",
        profile="diagnostic_test",
        step_deg=2,
        target_min_deg=0,
        target_max_deg=0,
    )

    assert energy_csv.is_file()
    assert torque_csv.is_file()
    assert not complete
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    assert metadata["torque_uncertainty_scope"] == "propagated_energy_uncertainty_only"
    assert metadata["finite_difference_truncation_error_bounded"] is False
    assert metadata["torque_numerically_certified"] is False
    with torque_csv.open(encoding="utf-8") as handle:
        fields = csv.DictReader(handle).fieldnames
    assert fields is not None
    assert "propagated_energy_error_bound_N_per_m" in fields
    assert "torque_error_bound_N_per_m" not in fields


def test_cleanup_removes_only_explicit_legacy_names(tmp_path: Path) -> None:
    legacy = tmp_path / "run_full_casimir_N896_scan.sh"
    keep = tmp_path / "important.py"
    legacy.write_text("legacy\n", encoding="utf-8")
    keep.write_text("keep\n", encoding="utf-8")

    removed = cleanup_legacy_root_scripts(root=tmp_path)

    assert removed == [legacy]
    assert not legacy.exists()
    assert keep.exists()


def test_normal_pilot_run_does_not_mutate_source_tree(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def unexpected_cleanup():
        raise AssertionError("normal energy runs must not delete source files")

    monkeypatch.setattr(workflow, "cleanup_legacy_root_scripts", unexpected_cleanup)
    monkeypatch.setattr(workflow, "_resources", lambda args: object())
    monkeypatch.setattr(workflow, "validate_pairings", lambda values: tuple(values))
    monkeypatch.setattr(workflow, "_energy_options", lambda args: object())

    def fake_migrate(args, pairings, resources, options, *, target_profile):
        seen["target_profile"] = target_profile

    monkeypatch.setattr(workflow, "_migrate", fake_migrate)
    monkeypatch.setattr(workflow, "run_energy_cases", lambda **kwargs: 0)

    assert workflow.main(["pilots", "--profile", "custom_pilot_profile"]) == 0
    assert seen["target_profile"] == "custom_pilot_profile"


def test_case_state_rejects_a_stale_configuration_before_skip(tmp_path: Path) -> None:
    run = tmp_path / "case"
    _write_completed_artifacts(run, config={"policy": "old"})

    assert _case_state(run) == "completed"
    assert _case_state(
        run,
        expected_config={"policy": "new"},
    ) == "configuration_mismatch"
    assert _case_state(
        run,
        expected_config={"policy": "old"},
    ) == "completed"


def test_corrupt_completed_result_is_resumed_not_skipped(tmp_path: Path) -> None:
    run = tmp_path / "case"
    _write_completed_artifacts(run, config={"policy": "current"})
    (run / "result.json").write_text("{truncated", encoding="utf-8")

    assert _case_state(
        run,
        expected_config={"policy": "current"},
    ) == "artifact_inconsistent"


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


def test_legacy_source_cache_may_differ_only_by_scheduling_fingerprint(
    tmp_path: Path,
) -> None:
    source_config = FixedCasimirConfig(
        pairings=("spm",),
        plate_angles_deg=(0.0, 0.0),
        logdet_rtol=1e-3,
        workers=30,
        parallel_mode="q",
        memory_budget_gb=0.0,
        max_context_workers=1,
    )
    target_config = FixedCasimirConfig(
        pairings=("spm",),
        plate_angles_deg=(0.0, 0.0),
        logdet_rtol=1.5e-3,
        workers=26,
        parallel_mode="q",
        memory_budget_gb=16.0,
        max_context_workers=1,
    )
    source_run = tmp_path / "source"
    source_cache = source_run / "cache" / "certified_points.json"
    source_cache.parent.mkdir(parents=True)
    (source_run / "config.json").write_text(
        json.dumps(
            {
                "outer_tail_config": {
                    "joint_config": {
                        "radial_config": {"point_config": source_config.as_dict()}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    legacy_policy = certified_point_policy_payload(
        source_config,
        frequency_extendable=True,
    )
    full_source = source_config.as_dict()
    for name in LEGACY_SCHEDULING_FIELDS:
        legacy_policy[name] = full_source[name]
    source_cache.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "policy_fingerprint": "legacy-scheduling-dependent-hash",
                "frequency_extendable": True,
                "point_policy": legacy_policy,
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    target_run = tmp_path / "target"
    report = migrate_cache(
        pairing="spm",
        source_run_dir=source_run,
        target_run_dir=target_run,
        target_point_config=target_config,
    )

    assert not report.skipped
    migrated = json.loads(
        (target_run / "cache" / "certified_points.json").read_text(encoding="utf-8")
    )
    assert migrated["policy_fingerprint"] == certified_point_policy_fingerprint(
        target_config,
        frequency_extendable=True,
    )


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
