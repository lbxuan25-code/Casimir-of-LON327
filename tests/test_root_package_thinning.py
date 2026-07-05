from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOVED_ROOT_FILES = (
    "bdg_q0_conventions.py",
    "finite_q_engine.py",
    "finite_q_quadrature.py",
    "gap_analysis.py",
    "normal_sampling.py",
)


def test_thinned_root_files_are_absent() -> None:
    for filename in MOVED_ROOT_FILES:
        assert not (ROOT / "src/lno327" / filename).exists()


def test_new_owner_imports_are_available() -> None:
    from lno327.analysis.gap import gap_statistics_by_band
    from lno327.analysis.normal_sampling import normal_sheet_tensor_from_sampling
    from lno327.diagnostics.bdg_q0_conventions import evaluate_bdg_q0_convention
    from lno327.workflows.finite_q_engine import FiniteQEngineOptions
    from lno327.workflows.finite_q_quadrature import FiniteQQuadratureOptions

    assert callable(gap_statistics_by_band)
    assert callable(normal_sheet_tensor_from_sampling)
    assert callable(evaluate_bdg_q0_convention)
    assert FiniteQEngineOptions().collective_mode == "amplitude_phase"
    assert FiniteQQuadratureOptions().integration_strategy == "best_available_adaptive"


def test_active_code_has_no_old_root_imports() -> None:
    old_modules = (
        "bdg_q0_conventions",
        "finite_q_engine",
        "finite_q_quadrature",
        "gap_analysis",
        "normal_sampling",
    )
    roots = (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "validation/lib",
        ROOT / "validation/scripts",
        ROOT / "scripts",
    )
    suffixes = {".py", ".md"}
    for base in roots:
        for path in base.rglob("*"):
            if path.suffix not in suffixes:
                continue
            text = path.read_text(encoding="utf-8")
            for module in old_modules:
                assert f"lno327.{module}" not in text
                assert f"from .{module}" not in text
