"""Production-only guards for the fixed transverse-point implementation."""
from __future__ import annotations

import ast
from pathlib import Path

from lno327.casimir import fixed_transverse_point_certification as certification
from lno327.casimir import fixed_transverse_point_engine as engine


def test_production_transverse_components_expose_required_surface() -> None:
    for value in (
        engine._build_context_jobs,
        engine._execute_level,
        engine.assess_frequency_level,
        certification.assess_frequency_level,
        certification.assess_oscillatory_envelope,
        certification.main,
    ):
        assert callable(value)


def test_production_parse_reserves_cpu_headroom(monkeypatch) -> None:
    monkeypatch.setattr(engine, "affinity_cpu_count", lambda: 32)
    monkeypatch.delenv("LNO327_CPU_RESERVE", raising=False)
    args = engine._parse_args(
        [
            "--q-point",
            "q",
            "0.01",
            "0.02",
            "--N-candidates",
            "128",
            "192",
            "256",
        ]
    )
    assert args.workers == 30
    assert args.worker_budget_source == "cpu_affinity_minus_reserved_headroom"


def test_transverse_production_modules_have_no_validation_imports() -> None:
    paths = (
        Path("src/lno327/casimir/fixed_transverse_point_engine.py"),
        Path("src/lno327/casimir/fixed_transverse_point_certification.py"),
    )
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "validation" or alias.name.startswith(
                        "validation."
                    ):
                        violations.append(f"{path}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "validation" or module.startswith("validation."):
                    violations.append(f"{path}:{node.lineno}:{module}")
    assert violations == []


def test_certification_uses_absolute_then_relative_policy() -> None:
    previous = {
        "shift_0": {"two_plate_logdet": -0.01, "hard_physical_passed": True},
        "shift_1": {"two_plate_logdet": -0.01, "hard_physical_passed": True},
    }
    current = {
        "shift_0": {"two_plate_logdet": -0.010005, "hard_physical_passed": True},
        "shift_1": {"two_plate_logdet": -0.010005, "hard_physical_passed": True},
    }
    result = certification.assess_frequency_level(
        current_by_shift=current,
        previous_by_shift=previous,
        rtol=1e-3,
        atol=1e-6,
    )
    assert result["accepted_transition"] is True
    assert result["adjacent_N_by_shift"]["shift_0"]["passed_by"] == "relative"
