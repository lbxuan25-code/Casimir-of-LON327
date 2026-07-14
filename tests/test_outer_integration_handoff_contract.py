from __future__ import annotations

from pathlib import Path

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


def test_outer_main_surface_has_no_positive_only_or_diagnostic_aliases():
    commands = set(available_commands())
    assert ("matsubara", "matsubara-orbit-gauss-crosscheck") in commands
    assert ("matsubara", "orbit-gauss-preflight") in commands
    assert ("matsubara", "total-orbit-gauss-scan") in commands
    assert ("matsubara", "positive-orbit-gauss-crosscheck") not in commands
    assert ("matsubara", "positive-orbit-gauss-scan") not in commands
    assert ("matsubara", "dwave-orbit-integrand-profile") not in commands
    assert ("diagnostic", "dwave-orbit-integrand-profile") in commands
    assert ("diagnostic", "dwave-diagonal-width-scan") in commands
