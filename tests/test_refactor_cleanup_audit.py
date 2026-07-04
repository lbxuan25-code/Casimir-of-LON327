import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "refactor_cleanup_audit.md"
SCRIPT = ROOT / "tools" / "audit_refactor_debt.py"

REQUIRED_SECTIONS = (
    "# Refactor Cleanup Audit",
    "## Executive Summary",
    "## Current Migration Status",
    "## Old Top-Level Module Inventory",
    "## Active Import Boundary Audit",
    "## Public API and __init__ Surface Audit",
    "## Performance and Repeated-Work Audit",
    "## Validation Scripts and Historical Outputs Audit",
    "## Test Suite Audit",
    "## Package Size and Consolidation Audit",
    "## Immediate Risks",
    "## Deletion Readiness Table",
    "## Recommended Cleanup Plan",
    "## Commands Run",
)

OLD_MODULE_NAMES = (
    "conductivity.py",
    "bdg_response.py",
    "nonlocal_response.py",
    "bdg_nonlocal_response.py",
    "finite_q_primitives.py",
    "tb_fourier.py",
    "ward_response.py",
    "ward_validation.py",
    "response_interface.py",
    "static_response.py",
    "response_conventions.py",
    "reflection_input.py",
    "casimir.py",
)

ACTIVE_SRC_EXCEPTIONS = {
    ROOT / "src" / "lno327" / "__init__.py",
    ROOT / "src" / "lno327" / "conductivity.py",
    ROOT / "src" / "lno327" / "bdg_response.py",
    ROOT / "src" / "lno327" / "finite_q_primitives.py",
    ROOT / "src" / "lno327" / "response_conventions.py",
    ROOT / "src" / "lno327" / "response_interface.py",
    ROOT / "src" / "lno327" / "static_response.py",
    ROOT / "src" / "lno327" / "ward_response.py",
    ROOT / "src" / "lno327" / "nonlocal_response.py",
    ROOT / "src" / "lno327" / "bdg_nonlocal_response.py",
    ROOT / "src" / "lno327" / "casimir.py",
}


def test_audit_script_runs_without_importing_package():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Refactor Debt Static Audit Summary" in result.stdout
    assert "Old module references" in result.stdout
    assert "Top expensive-pattern files" in result.stdout


def test_refactor_cleanup_audit_report_contains_required_sections_and_old_modules():
    text = REPORT.read_text()

    for section in REQUIRED_SECTIONS:
        assert section in text
    for module_name in OLD_MODULE_NAMES:
        assert module_name in text


def test_refactor_cleanup_audit_does_not_promote_response_to_casimir_ready():
    text = REPORT.read_text()

    assert "valid_for_casimir_input=False" in text
    assert "Current response objects still correctly carry `valid_for_casimir_input=False`" in text
    assert "does not promote any response path to Casimir-ready status" in text


def test_active_new_src_no_longer_imports_selected_old_top_level_modules():
    forbidden = (
        "from .conductivity",
        "from .bdg_response",
        "from .finite_q_primitives",
        "from .response_conventions",
        "from lno327.conductivity",
        "from lno327.bdg_response",
        "from lno327.finite_q_primitives",
        "from lno327.response_conventions",
        "import lno327.conductivity",
        "import lno327.bdg_response",
        "import lno327.finite_q_primitives",
        "import lno327.response_conventions",
    )

    for path in (ROOT / "src" / "lno327").rglob("*.py"):
        if path in ACTIVE_SRC_EXCEPTIONS:
            continue
        text = path.read_text()
        for needle in forbidden:
            assert needle not in text, f"{path} should not contain {needle!r}"
