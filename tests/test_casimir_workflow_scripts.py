from __future__ import annotations

from pathlib import Path
import math

import pytest

from scripts.casimir.cleanup_legacy_root import cleanup_legacy_root_scripts
from scripts.casimir.config import (
    angle_token,
    case_name,
    inclusive_integer_grid,
    select_runtime_resources,
)
from scripts.casimir.postprocess import (
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
        "spm_T10K_d20nm_theta_p000deg_runtime_budget_v2"
    )


def test_cpu_selection_reserves_requested_logical_capacity() -> None:
    resources = select_runtime_resources(
        available_cpus=tuple(range(32)),
        reserve_logical_cpus=4,
        worker_cap=28,
    )
    assert resources.workers == 28
    assert len(resources.reserved_cpus) == 4
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
