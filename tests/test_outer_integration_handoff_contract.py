from __future__ import annotations

from pathlib import Path

from lno327.casimir import FixedCasimirConfig, run_casimir
from validation.__main__ import available_commands


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "docs/full_outer_integration_handoff.md"
RETIRED_PIPELINE = ROOT / "scripts/casimir/finite_q_bdg_casimir_pipeline.py"
CASIMIR_README = ROOT / "scripts/casimir/README.md"


def test_retired_monolithic_outer_pipeline_is_absent():
    assert not RETIRED_PIPELINE.exists()
    assert CASIMIR_README.is_file()


def test_outer_handoff_preserves_hard_readiness_state():
    text = HANDOFF.read_text(encoding="utf-8")
    for statement in (
        "diagnostic_only = True",
        "production_reference_established = False",
        "valid_for_casimir_input = False",
        "passive_sheet_logdet",
        "exact-diagonal variant sensitivity",
        "sigma=-K/xi",
    ):
        assert statement in text


def test_outer_main_surface_uses_production_fixed_controller():
    assert callable(run_casimir)
    assert FixedCasimirConfig().matsubara_indices == (0, 1)
    commands = set(available_commands())
    assert ("matsubara", "matsubara-orbit-gauss-crosscheck") in commands
    assert ("matsubara", "orbit-gauss-preflight") in commands
    assert ("matsubara", "total-orbit-gauss-scan") in commands
    assert ("diagnostic", "transverse-point-sweet-spot") not in commands
    assert ("casimir", "microscopic-outer-q-preflight") not in commands
    assert ("matsubara", "positive-orbit-gauss-crosscheck") not in commands
    assert ("matsubara", "positive-orbit-gauss-scan") not in commands
    for retired in (
        ("matsubara", "dwave-orbit-integrand-profile"),
        ("diagnostic", "dwave-orbit-integrand-profile"),
        ("diagnostic", "dwave-diagonal-width-scan"),
        ("diagnostic", "dwave-orbit-adaptive"),
    ):
        assert retired not in commands
