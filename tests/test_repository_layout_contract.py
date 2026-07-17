from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_retired_repository_trees_are_absent() -> None:
    retired = (
        "results",
        "sandbox",
        "validation/scripts",
        "outputs/casimir/finite_q_bdg_pipeline",
    )
    for relative in retired:
        assert not (ROOT / relative).exists(), relative


def test_active_validation_surface_is_grouped() -> None:
    validation_root = ROOT / "validation"
    assert (validation_root / "__main__.py").is_file()
    assert (validation_root / "commands/ward").is_dir()
    assert (validation_root / "commands/static").is_dir()
    assert (validation_root / "commands/matsubara").is_dir()
    assert (validation_root / "lib/finite_q_validation_models.py").is_file()


def test_fixed_casimir_validation_routes_and_legacy_files_are_absent() -> None:
    validation_root = ROOT / "validation"
    for relative in (
        "commands/matsubara/transverse_point_sweet_spot.py",
        "commands/casimir/microscopic_outer_q_preflight.py",
        "lib/matsubara.py",
        "lib/microscopic_outer_q_compound.py",
        "lib/microscopic_outer_q_preflight.py",
        "lib/microscopic_outer_q_preflight_legacy.py",
        "lib/transverse_point_sweet_spot_engine.py",
        "lib/transverse_point_sweet_spot_command.py",
        "lib/transverse_point_sweet_spot_engine_legacy.py",
    ):
        assert not (validation_root / relative).exists(), relative


def test_superseded_point_convergence_files_are_absent() -> None:
    validation_root = ROOT / "validation"
    for relative in (
        "commands/static/nk_scan.py",
        "commands/static/dwave_gauss_outer.py",
        "commands/static/dwave_orbit_gauss.py",
        "commands/static/projection_scan.py",
        "commands/static/quadrature_compare.py",
        "commands/matsubara/positive_point.py",
        "commands/matsubara/arbitrary_q_uniform_refinement_diagnostic.py",
        "commands/matsubara/dwave_small_xi.py",
        "commands/matsubara/bond_metric_positive.py",
        "commands/matsubara/dwave_orbit_adaptive.py",
        "commands/matsubara/dwave_orbit_panel_adaptive.py",
        "commands/matsubara/dwave_orbit_evaluator_profile.py",
        "commands/matsubara/dwave_orbit_integrand_profile.py",
        "commands/matsubara/dwave_diagonal_width_scan.py",
        "commands/matsubara/dwave_orbit_gauss_crosscheck.py",
        "commands/matsubara/dwave_orbit_certification_scan.py",
        "commands/matsubara/dwave_orbit_certification_scan_parallel.py",
        "lib/static_point_diagnostics.py",
    ):
        assert not (validation_root / relative).exists(), relative


def test_validation_root_has_no_bare_runner_or_analyzer_modules() -> None:
    root_modules = {path.name for path in (ROOT / "validation").glob("*.py")}
    assert root_modules == {"__init__.py", "__main__.py"}


def test_validation_outputs_are_ignored_as_runtime_artifacts() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "validation/outputs/**" in gitignore
    assert "!validation/outputs" not in gitignore
