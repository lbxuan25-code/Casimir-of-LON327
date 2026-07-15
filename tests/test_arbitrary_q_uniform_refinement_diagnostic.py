from __future__ import annotations

from pathlib import Path

import numpy as np

from validation.__main__ import resolve_command
from validation.commands.matsubara.arbitrary_q_uniform_refinement_diagnostic import (
    block_resolved_primitive_change,
    primitive_block_slices,
)


def test_uniform_refinement_block_layout_covers_complete_primitive() -> None:
    groups = primitive_block_slices(2)
    assert groups[0][0] == "direct_contact"
    assert groups[-1][0] == "n_index_1:collective_em_right"
    assert groups[-1][1].stop == 18 + 2 * 25


def test_block_resolved_change_identifies_worst_physical_block() -> None:
    previous = np.zeros(43, dtype=complex)
    current = np.zeros(43, dtype=complex)
    previous[0] = current[0] = 1.0
    current[18 + 9] = 1e-2
    result = block_resolved_primitive_change(
        previous,
        current,
        frequency_count=1,
        rtol=1e-3,
        atol=1e-12,
    )
    assert result["passed"] is False
    assert result["worst_block"] == "n_index_0:collective_bubble"
    assert result["max_mixed_ratio"] > 1.0


def test_uniform_refinement_command_is_the_only_extra_arbitrary_q_diagnostic() -> None:
    assert resolve_command("diagnostic", "arbitrary-q-uniform-refinement") == (
        "validation.commands.matsubara.arbitrary_q_uniform_refinement_diagnostic"
    )


def test_retained_arbitrary_q_workflow_surface_is_allowlisted() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow_dir = root / "src" / "lno327" / "workflows"
    actual = {path.name for path in workflow_dir.glob("arbitrary_q_*.py")}
    assert actual == {
        "arbitrary_q_matsubara.py",
        "arbitrary_q_parallel.py",
    }


def test_retained_arbitrary_q_validation_surface_is_allowlisted() -> None:
    root = Path(__file__).resolve().parents[1]
    command_dir = root / "validation" / "commands" / "matsubara"
    actual = {path.name for path in command_dir.glob("arbitrary_q_*.py")}
    assert actual == {
        "arbitrary_q_performance_preflight.py",
        "arbitrary_q_performance_smoke.py",
        "arbitrary_q_periodic_bz_qualification.py",
        "arbitrary_q_periodic_bz_qualification_gate.py",
        "arbitrary_q_physics_smoke.py",
        "arbitrary_q_uniform_refinement_diagnostic.py",
    }
