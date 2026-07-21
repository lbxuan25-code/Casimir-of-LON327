from __future__ import annotations

import lno327.casimir as casimir
from lno327.casimir.legacy import FixedCasimirConfig, run_fixed_reference_casimir
from lno327.casimir.production import (
    FullCasimirConfig,
    build_full_casimir_config,
    run_full_casimir,
)


def test_package_root_exposes_internal_full_engine_without_competing_controllers() -> None:
    assert casimir.build_full_casimir_config is build_full_casimir_config
    assert casimir.run_full_casimir is run_full_casimir
    assert casimir.FullCasimirConfig is FullCasimirConfig
    for retired_name in (
        "run_casimir",
        "FixedCasimirConfig",
        "AdaptiveRadialCasimirConfig",
        "run_adaptive_radial_casimir",
        "AdaptiveAngularCasimirConfig",
        "run_adaptive_angular_casimir",
        "AdaptiveJointCasimirConfig",
        "run_adaptive_joint_casimir",
        "AdaptiveOuterTailCasimirConfig",
        "run_adaptive_outer_tail_casimir",
        "AdaptiveMatsubaraCasimirConfig",
        "run_adaptive_matsubara_casimir",
    ):
        assert not hasattr(casimir, retired_name), retired_name


def test_fixed_reference_library_is_isolated_from_operational_command() -> None:
    assert callable(run_fixed_reference_casimir)
    assert FixedCasimirConfig().matsubara_indices == (0, 1)
    metadata = casimir.casimir_layer_metadata()
    assert metadata["canonical_operational_entrypoint"] == (
        "python -m scripts.full_casimir"
    )
    assert metadata["package_command_present"] is False
    assert metadata["installed_console_command_present"] is False
    assert metadata["legacy_calculation_scripts_present"] is False


def test_canonical_builder_wires_external_physical_inputs() -> None:
    config = build_full_casimir_config(
        pairings=("spm", "dwave"),
        temperature_K=12.0,
        separation_nm=25.0,
        plate_angles_deg=(3.0, 29.0),
        matsubara_cutoff_values=(1, 3, 7, 15),
        cutoff_u_values=(6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0),
        point_cache_path="outputs/casimir/production/test/runs/case/cache/certified_points.json",
    )
    point = config.outer_tail_config.joint_config.radial_config.point_config
    assert point.pairings == ("spm", "dwave")
    assert point.temperature_K == 12.0
    assert point.separation_nm == 25.0
    assert point.plate_angles_deg == (3.0, 29.0)
    assert config.matsubara_cutoff_values == (1, 3, 7, 15)
    assert config.outer_tail_config.cutoff_u_values[-3:] == (30.0, 36.0, 42.0)
    assert config.point_cache_path.name == "certified_points.json"
