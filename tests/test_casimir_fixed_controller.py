"""Contract tests for the unique fixed production Casimir controller."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lno327.casimir import fixed_chain
from lno327.casimir.fixed_chain import FixedCasimirConfig, run_casimir
from lno327.casimir.fixed_outer_q import (
    build_staged_grid_plan,
    build_union_node_manifest,
)
from lno327.constants import KB
from lno327.electrodynamics.materials import LNO327_THIN_FILM_SLAO_IN_PLANE


def _point(pairing: str, label: str, n: int, value: float) -> dict:
    shifts = {
        f"shift_{index}:{shift}": {
            "two_plate_logdet": float(value),
            "hard_physical_passed": True,
        }
        for index, shift in enumerate(fixed_chain.DEFAULT_SHIFTS)
    }
    return {
        "pairing": pairing,
        "q_label": label,
        "n": int(n),
        "sweet_spot": {
            "status": "established",
            "working_N": 192,
            "audit_N": 256,
            "establishment_mode": "strict_consecutive_adjacent",
        },
        "history": [
            {
                "N": 256,
                "shifts": shifts,
                "two_plate_logdet_cross_shift": {"passed": True},
            }
        ],
    }


def _payload(config: FixedCasimirConfig, manifest, values_by_n: dict[int, float]) -> dict:
    return {
        "schema": "transverse-point-sweet-spot-v4",
        "run_complete": True,
        "all_requested_sweet_spots_established": True,
        "point_results": [
            _point(pairing, label, n, values_by_n[n])
            for pairing in config.pairings
            for label in manifest.labels
            for n in config.matsubara_indices
        ],
    }


def _mock_certifier(payload_builder):
    def run(config, manifest, output):
        payload = payload_builder(config, manifest)
        return fixed_chain._CertificationRun(
            payload=payload,
            stdout="{}",
            stderr="",
            command=("python", "-m", "lno327.casimir.fixed_transverse_point_certification"),
        )

    return run


def test_run_casimir_composes_certifier_and_fixed_outer_reducer(monkeypatch) -> None:
    config = FixedCasimirConfig(
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
        separation_nm=1e9,
        outer_rtol=0.0,
        outer_atol_J_m2=1e9,
        workers=1,
        parallel_mode="serial",
    )
    seen = {}

    def payload_builder(current, manifest):
        seen["labels"] = manifest.labels
        seen["q_model"] = np.array(manifest.q_model, copy=True)
        return _payload(current, manifest, {0: 2.0, 1: 3.0})

    monkeypatch.setattr(
        fixed_chain,
        "_run_transverse_certifier",
        _mock_certifier(payload_builder),
    )
    result = run_casimir(config)

    assert result.status == "finite_partial"
    assert result.production_casimir_allowed is False
    assert result.partial_sum_only is True
    assert result.matsubara_tail_estimated is False
    assert result.all_microscopic_nodes_certified is True
    assert result.finite_partial_outer_q_integrals_available is True
    assert result.candidate_outer_q_budget_established is True
    assert len(seen["labels"]) == result.grid_plan["unique_microscopic_q_node_count"]
    assert seen["q_model"].shape == (len(seen["labels"]), 2)

    reference = result.reference_results["spm"]
    exact_measure = 2.0**2 / (16.0 * np.pi * 1.0**2)
    expected = KB * config.temperature_K * exact_measure * np.asarray([1.0, 3.0])
    np.testing.assert_allclose(reference["contributions_J_m2"], expected, rtol=1e-14)
    np.testing.assert_allclose(
        reference["partial_free_energy_J_m2"],
        float(np.sum(expected)),
        rtol=1e-14,
    )
    assert reference["minimum_working_N"] == 192
    assert reference["maximum_audit_N"] == 256


def test_run_casimir_fails_closed_when_a_microscopic_point_is_missing(monkeypatch) -> None:
    config = FixedCasimirConfig(
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
        workers=1,
        parallel_mode="serial",
    )

    def payload_builder(current, manifest):
        payload = _payload(current, manifest, {0: 2.0, 1: 3.0})
        payload["point_results"] = payload["point_results"][1:]
        payload["all_requested_sweet_spots_established"] = False
        return payload

    monkeypatch.setattr(
        fixed_chain,
        "_run_transverse_certifier",
        _mock_certifier(payload_builder),
    )
    result = run_casimir(config)

    assert result.status == "unresolved"
    assert result.production_casimir_allowed is False
    assert result.all_microscopic_nodes_certified is False
    assert result.finite_partial_outer_q_integrals_available is False
    assert result.candidate_outer_q_budget_established is False
    assert result.unresolved_microscopic_points


def test_controller_invokes_only_the_production_certification_module(tmp_path) -> None:
    config = FixedCasimirConfig(
        u_max_values=(1.0, 2.0),
        radial_orders=(1, 2),
        angular_orders=(2, 4),
        angular_offsets=(0.0, 0.5),
    )
    plan = build_staged_grid_plan(
        u_max_values=config.u_max_values,
        radial_orders=config.radial_orders,
        angular_orders=config.angular_orders,
        angular_offsets=config.angular_offsets,
    )
    material = LNO327_THIN_FILM_SLAO_IN_PLANE
    manifest = build_union_node_manifest(
        plan,
        separation_m=config.separation_m,
        lattice_a_x_m=material.lattice_a_x_m,
        lattice_a_y_m=material.lattice_a_y_m,
    )
    command = fixed_chain._transverse_certification_command(
        config,
        manifest,
        tmp_path / "certification.json",
    )

    assert command[1:3] == [
        "-m",
        "lno327.casimir.fixed_transverse_point_certification",
    ]
    assert all("validation" not in value for value in command)
    assert command.count("--q-point") == len(manifest.labels)


def test_controller_replays_the_qualified_spm_n01_reference_contract(monkeypatch) -> None:
    reference_path = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "references"
        / "casimir"
        / "spm_n01_outer_q_reference_v1.json"
    )
    qualified = json.loads(reference_path.read_text(encoding="utf-8"))
    target_integrals = qualified["reference_result"]["outer_q_integrals_m_inv2"]
    config = FixedCasimirConfig(outer_atol_J_m2=1e9)

    def payload_builder(current, manifest):
        plan = build_staged_grid_plan(
            u_max_values=current.u_max_values,
            radial_orders=current.radial_orders,
            angular_orders=current.angular_orders,
            angular_offsets=current.angular_offsets,
        )
        grid = manifest.grids[plan.reference_spec_id]
        measure = float(np.sum(grid.measure_weights_m_inv2))
        values = {
            n: float(target_integrals[position]) / measure
            for position, n in enumerate(current.matsubara_indices)
        }
        return _payload(current, manifest, values)

    monkeypatch.setattr(
        fixed_chain,
        "_run_transverse_certifier",
        _mock_certifier(payload_builder),
    )
    result = run_casimir(config)
    actual = result.reference_results["spm"]
    expected = qualified["reference_result"]

    assert result.status == "finite_partial"
    assert result.reference_spec_id == qualified["outer_q_contract"]["reference_spec_id"]
    np.testing.assert_allclose(
        actual["outer_q_integrals_m_inv2"],
        expected["outer_q_integrals_m_inv2"],
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        actual["contributions_J_m2"],
        expected["contributions_J_m2"],
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        actual["partial_free_energy_J_m2"],
        expected["partial_free_energy_J_m2"],
        rtol=1e-14,
    )
    assert actual["minimum_working_N"] == qualified["microscopic_contract"]["working_N"]
    assert actual["maximum_audit_N"] == qualified["microscopic_contract"]["audit_N"]
    assert result.production_casimir_allowed is False
