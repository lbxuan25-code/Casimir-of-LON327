"""Tests for the clean geometry-free TODO 2 response ladder engine."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lno327.casimir import material_response as material
from lno327.casimir import material_response_engine as engine
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
)


def _sample(
    *,
    q: np.ndarray,
    xi_eV: float,
    value: float,
    fingerprint: str,
) -> material.MaterialResponseSample:
    validation = StaticSheetValidation(
        finite=True,
        ward_passed=True,
        relative_imaginary_norm=0.0,
        relative_longitudinal_gauge_residual=0.0,
        relative_density_transverse_mixing=0.0,
        chi_bar=value,
        dbar_t=2.0 * value,
        reality_tolerance=1e-8,
        longitudinal_tolerance=1e-6,
        mixing_tolerance=1e-6,
        passivity_tolerance=1e-10,
    )
    response = StaticSheetResponse(
        kernel_lt=np.eye(3, dtype=complex),
        chi_bar=value,
        dbar_t=2.0 * value,
        q_model=q,
        energy_scale_eV=1.0,
        degeneracy=1.0,
        basis=STATIC_LOCAL_BASIS,
        validation=validation,
        metadata={"source": "test"},
    )
    side = SimpleNamespace(effective_mixed_ratio=0.0)
    ward = SimpleNamespace(
        passed=True,
        left=side,
        right=side,
        schur_condition_number=1.0,
    )
    return material.MaterialResponseSample(
        frequency_index=0,
        frequency_sector="zero_matsubara",
        q_crystal=q,
        xi_eV=0.0,
        material_cache_fingerprint=fingerprint,
        kernel=SimpleNamespace(q_model=q, xi_eV=0.0),
        operator_ward=SimpleNamespace(passed=True),
        effective_ward=ward,
        strict_static_ward=SimpleNamespace(passed=True),
        response=response,
        sheet_validation=validation,
        metadata={},
    )


def test_engine_module_has_no_geometry_dependencies() -> None:
    path = Path("src/lno327/casimir/material_response_engine.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = (
        "lno327.casimir.lifshitz",
        "lno327.casimir.outer",
        "lno327.electrodynamics.reflection",
        "lno327.casimir.material_geometry",
    )
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for module_name in modules:
            if any(
                module_name == prefix or module_name.startswith(prefix + ".")
                for prefix in forbidden
            ):
                violations.append(f"{node.lineno}:{module_name}")
    assert violations == []


def test_engine_config_payload_is_geometry_free() -> None:
    config = engine.MaterialResponseEngineConfig(pairing_name="spm")
    payload = config.as_dict()
    assert payload["q_input_basis"] == "crystal_xy"
    assert payload["geometry_inputs_present"] is False
    assert payload["production_casimir_allowed"] is False
    assert not {
        "q_lab",
        "theta_rad",
        "plate_angles_rad",
        "separation_nm",
        "outer_order",
    }.intersection(payload)


def test_engine_batches_frequencies_and_certifies_without_geometry(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeModel:
        spec = object()

        def build_ansatz(self, pairing_name, *, phase_vertex):
            assert pairing_name == "spm"
            assert phase_vertex == "bond_endpoint_gauge"
            return object()

        def build_pairing_params(self, delta0_eV):
            assert delta0_eV == 0.1
            return object()

    monkeypatch.setattr(
        engine,
        "get_finite_q_microscopic_model",
        lambda name: FakeModel(),
    )

    def fake_integrate(**kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace(
            q_model=np.array(kwargs["q_model"], dtype=float, copy=True),
            xi_eV_values=np.array(kwargs["xi_eV_values"], dtype=float, copy=True),
            components=tuple(object() for _ in kwargs["xi_eV_values"]),
            rhs=tuple(object() for _ in kwargs["xi_eV_values"]),
            material_cache_fingerprint=(
                f"N{kwargs['n']}:shift{tuple(kwargs['shift'])}"
            ),
            operator_ward=SimpleNamespace(passed=True),
            metadata={"N": kwargs["n"], "shift": tuple(kwargs["shift"])},
        )

    monkeypatch.setattr(engine, "integrate_arbitrary_q_periodic_bz", fake_integrate)

    def fake_builder(result, *, frequency_index, policy):
        _ = policy
        n_grid = int(result.metadata["N"])
        shift = tuple(result.metadata["shift"])
        shift_offset = 0.0002 if shift == (0.25, 0.75) else 0.0
        frequency_offset = 0.0001 * int(frequency_index)
        value = 1.0 + 1.0 / n_grid + shift_offset + frequency_offset
        return _sample(
            q=np.asarray(result.q_model, dtype=float),
            xi_eV=float(result.xi_eV_values[frequency_index]),
            value=value,
            fingerprint=result.material_cache_fingerprint,
        )

    monkeypatch.setattr(engine, "build_material_response_sample", fake_builder)
    monkeypatch.setattr(
        engine,
        "matsubara_energy_eV",
        lambda index, temperature: 0.0,
    )

    config = engine.MaterialResponseEngineConfig(
        pairing_name="spm",
        matsubara_indices=(0, 1),
        n_candidates=(64, 96, 128),
        shifts=((0.5, 0.5), (0.25, 0.75)),
        required_consecutive_passes=2,
        convergence_policy=engine.MaterialResponseConvergencePolicy(
            relative_tolerance=2e-2,
            absolute_tolerance=1e-8,
        ),
    )
    q = np.array([0.01, np.nextafter(0.02, 1.0)])
    result = engine.evaluate_material_response_ladder(config, q_crystal=q)

    assert result.all_requested_certified is True
    assert result.evaluated_n_candidates == (64, 96, 128)
    assert np.array_equal(result.q_crystal, q)
    assert result.q_crystal.flags.writeable is False
    assert result.metadata["geometry_inputs_present"] is False
    assert result.metadata["reflection_constructed"] is False
    assert result.metadata["two_plate_logdet_constructed"] is False
    assert result.metadata["production_casimir_allowed"] is False
    assert result.frequencies[0].certification.primary_response.q_crystal.tolist() == q.tolist()

    assert len(calls) == 6
    assert all(np.array_equal(call["q_model"], q) for call in calls)
    assert all(len(call["xi_eV_values"]) == 2 for call in calls)
    for call in calls:
        assert not {
            "q_lab",
            "theta_rad",
            "plate_angles_rad",
            "separation_nm",
            "outer_order",
        }.intersection(call)
