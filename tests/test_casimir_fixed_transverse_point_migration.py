"""Guards for fixed transverse-point production migration."""
from __future__ import annotations

import ast
from pathlib import Path


def test_validation_engine_surface_resolves_to_production() -> None:
    from lno327.casimir import fixed_transverse_point_engine as production
    from validation.lib import transverse_point_sweet_spot_engine as facade

    for name in (
        "_build_context_jobs",
        "_execute_level",
        "_parse_args",
        "assess_frequency_level",
    ):
        assert getattr(facade, name) is getattr(production, name)


def test_validation_command_uses_production_controller() -> None:
    from lno327.casimir import (
        fixed_transverse_point_certification as production,
    )
    from validation.lib import transverse_point_sweet_spot_command as facade

    assert facade.DEFAULT_LOGDET_ATOL == production.DEFAULT_LOGDET_ATOL
    assert facade.ENVELOPE_LEVELS == production.ENVELOPE_LEVELS
    assert facade._production is production


def test_new_production_modules_have_no_validation_imports() -> None:
    paths = (
        Path("src/lno327/casimir/fixed_transverse_point_engine.py"),
        Path(
            "src/lno327/casimir/"
            "fixed_transverse_point_certification.py"
        ),
    )
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "validation" or alias.name.startswith(
                        "validation."
                    ):
                        violations.append(
                            f"{path}:{node.lineno}:{alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "validation" or module.startswith(
                    "validation."
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{module}"
                    )
    assert violations == []


def test_production_engine_keeps_legacy_numerical_source_body() -> None:
    legacy = Path(
        "validation/lib/transverse_point_sweet_spot_engine_legacy.py"
    ).read_text(encoding="utf-8")
    production = Path(
        "src/lno327/casimir/fixed_transverse_point_engine.py"
    ).read_text(encoding="utf-8")

    assert production.count("def _plate_state(") == legacy.count(
        "def _plate_state("
    )
    assert production.count("def _two_plate_state(") == legacy.count(
        "def _two_plate_state("
    )
    assert production.count("def _execute_level(") == legacy.count(
        "def _execute_level("
    )
    assert production.count("def assess_frequency_level(") == legacy.count(
        "def assess_frequency_level("
    )
    assert "get_finite_q_microscopic_model(" in production
    assert "get_finite_q_validation_model(" not in production
