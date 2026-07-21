from __future__ import annotations

from pathlib import Path
import csv
import json
import math

import pytest

from scripts.full_casimir import __main__ as unified
from scripts.full_casimir.config import (
    angle_token,
    inclusive_integer_grid,
    physical_case_name,
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
    reason = "outer_and_matsubara_certificates_and_total_budget_met"
    common = {
        "selected_matsubara_cutoff": 1,
        "production_casimir_allowed": True,
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
                "production_casimir_allowed": True,
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


def test_angle_grid_and_physical_case_names_are_deterministic() -> None:
    assert inclusive_integer_grid(-4, 94, 2)[0] == -4
    assert inclusive_integer_grid(-4, 94, 2)[-1] == 94
    assert len(inclusive_integer_grid(-4, 94, 2)) == 50
    assert angle_token(-4) == "m004"
    assert angle_token(0) == "p000"
    assert angle_token(94) == "p094"
    assert physical_case_name("spm", 0) == "spm_T10K_d20nm_theta_p000deg"


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


def test_case_state_rejects_stale_configuration_before_skip(tmp_path: Path) -> None:
    run = tmp_path / "case"
    _write_completed_artifacts(run, config={"policy": "old"})
    assert _case_state(run) == "completed"
    assert _case_state(run, expected_config={"policy": "new"}) == (
        "configuration_mismatch"
    )
    assert _case_state(run, expected_config={"policy": "old"}) == "completed"


def test_corrupt_completed_result_is_not_skipped(tmp_path: Path) -> None:
    run = tmp_path / "case"
    _write_completed_artifacts(run, config={"policy": "current"})
    (run / "result.json").write_text("{truncated", encoding="utf-8")
    assert _case_state(run, expected_config={"policy": "current"}) == (
        "artifact_inconsistent"
    )


def test_top_level_dispatcher_has_no_legacy_calculation_aliases() -> None:
    assert "legacy-workflow" not in unified._COMMANDS
    assert "qualification" not in unified._COMMANDS
    assert "pilots" not in unified._COMMANDS
    assert "scan" not in unified._COMMANDS
    assert "all" not in unified._COMMANDS
