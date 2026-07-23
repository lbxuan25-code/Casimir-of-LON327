"""Boundary tests for geometry-independent material response construction."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lno327.casimir import material_response as material
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetConductivityConversion,
    SheetResponseValidation,
)
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
)


def _operator_ward(*, passed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(passed=passed)


def _effective_ward(*, passed: bool = True) -> SimpleNamespace:
    side = SimpleNamespace(effective_mixed_ratio=0.25)
    return SimpleNamespace(
        passed=passed,
        left=side,
        right=side,
        schur_condition_number=2.0,
    )


def _kernel(q: np.ndarray, xi_eV: float) -> SimpleNamespace:
    return SimpleNamespace(q_model=np.asarray(q, dtype=float), xi_eV=float(xi_eV))


def _static_sheet(q: np.ndarray) -> StaticSheetResponse:
    validation = StaticSheetValidation(
        finite=True,
        ward_passed=True,
        relative_imaginary_norm=0.0,
        relative_longitudinal_gauge_residual=2e-7,
        relative_density_transverse_mixing=0.0,
        chi_bar=1.5,
        dbar_t=0.75,
        reality_tolerance=1e-8,
        longitudinal_tolerance=1e-6,
        mixing_tolerance=1e-6,
        passivity_tolerance=1e-10,
    )
    return StaticSheetResponse(
        kernel_lt=np.eye(3, dtype=complex),
        chi_bar=1.5,
        dbar_t=0.75,
        q_model=q,
        energy_scale_eV=1.0,
        degeneracy=1.0,
        basis=STATIC_LOCAL_BASIS,
        validation=validation,
        metadata={"source": "test"},
    )


def _positive_sheet(q: np.ndarray, xi_eV: float) -> PositiveMatsubaraSheetResponse:
    model_tensor = ConductivityTensor(1.0, 2.0, 0.1, 0.1)
    sheet = SheetConductivityConversion(
        tensor=ConductivityTensor(3.0, 4.0, 0.2, 0.2),
        unit_stage="sheet_conductivity",
        unit_label="test_sheet",
        normalization_status="test",
        valid_for_casimir_input=True,
        notes=(),
    )
    tilde = SheetConductivityConversion(
        tensor=ConductivityTensor(0.3, 0.4, 0.02, 0.02),
        unit_stage="reflection_dimensionless_conductivity",
        unit_label="test_tilde",
        normalization_status="test",
        valid_for_casimir_input=True,
        notes=(),
    )
    return PositiveMatsubaraSheetResponse(
        sigma_model_xy=model_tensor,
        sigma_sheet_si_xy=sheet,
        sigma_tilde_xy=tilde,
        q_model=q,
        xi_eV=xi_eV,
        degeneracy=1.0,
        basis="crystal_xy",
        metadata={"source": "test"},
    )


def _positive_validation() -> SheetResponseValidation:
    return SheetResponseValidation(
        finite=True,
        relative_imaginary_norm=0.0,
        relative_symmetry_residual=0.0,
        minimum_symmetric_eigenvalue=0.1,
        reality_tolerance=1e-9,
        symmetry_tolerance=1e-9,
        passivity_tolerance=1e-10,
    )


def test_policy_is_frozen_and_geometry_free() -> None:
    policy = material.MaterialResponsePolicy()
    payload = policy.as_dict()
    assert payload["schema"] == "material-response-policy-v1"
    assert not {
        "q_lab",
        "theta_rad",
        "plate_angles_rad",
        "separation_nm",
        "outer_order",
    }.intersection(payload)
    with pytest.raises(FrozenInstanceError):
        policy.degeneracy = 2.0


def test_material_response_module_has_no_geometry_imports() -> None:
    path = Path("src/lno327/casimir/material_response.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = (
        "lno327.casimir.lifshitz",
        "lno327.casimir.outer",
        "lno327.electrodynamics.reflection",
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


def test_static_sample_is_readonly_and_keeps_geometry_out() -> None:
    q = np.array([0.01, -0.02])
    sheet = _static_sheet(q)
    sample = material.MaterialResponseSample(
        frequency_index=0,
        frequency_sector="zero_matsubara",
        q_crystal=q,
        xi_eV=0.0,
        material_cache_fingerprint="material-test",
        kernel=_kernel(q, 0.0),
        operator_ward=_operator_ward(),
        effective_ward=_effective_ward(),
        strict_static_ward=SimpleNamespace(passed=True),
        response=sheet,
        sheet_validation=sheet.validation,
        metadata={"casimir_stage": "geometry_independent_material_response"},
    )

    assert sample.hard_physical_passed is True
    assert sample.primary_matrix.flags.writeable is False
    assert sample.q_crystal.flags.writeable is False
    assert np.allclose(np.diag(sample.primary_matrix), [1.5, 0.75])
    diagnostics = sample.diagnostics()
    assert diagnostics["frequency_sector"] == "zero_matsubara"
    assert diagnostics["material_hard_physical_passed"] is True
    assert diagnostics["static_longitudinal_warning"] is False
    assert not {"q_lab", "theta_rad", "separation_nm"}.intersection(diagnostics)

    with pytest.raises(ValueError):
        sample.q_crystal[0] = 1.0
    with pytest.raises(TypeError):
        sample.metadata["new"] = "value"


def test_sector_payload_mismatch_fails_at_construction() -> None:
    q = np.array([0.01, 0.02])
    positive = _positive_sheet(q, 0.1)
    with pytest.raises(TypeError, match="StaticSheetResponse"):
        material.MaterialResponseSample(
            frequency_index=0,
            frequency_sector="zero_matsubara",
            q_crystal=q,
            xi_eV=0.0,
            material_cache_fingerprint="material-test",
            kernel=_kernel(q, 0.0),
            operator_ward=_operator_ward(),
            effective_ward=_effective_ward(),
            strict_static_ward=SimpleNamespace(passed=True),
            response=positive,
            sheet_validation=_positive_validation(),
            metadata={},
        )


def test_builder_uses_only_static_path_for_exact_zero(monkeypatch) -> None:
    q = np.array([0.03, -0.01])
    sheet = _static_sheet(q)
    kernel = _kernel(q, 0.0)
    ward = _effective_ward()
    result = SimpleNamespace(
        components=(object(),),
        rhs=(object(),),
        q_model=q,
        xi_eV_values=np.array([0.0]),
        material_cache_fingerprint="material-static",
        operator_ward=_operator_ward(),
        metadata={},
    )

    monkeypatch.setattr(
        material,
        "effective_em_kernel_from_components",
        lambda *args, **kwargs: kernel,
    )
    monkeypatch.setattr(
        material,
        "validate_effective_ward_xy",
        lambda *args, **kwargs: ward,
    )
    monkeypatch.setattr(
        material,
        "validate_strict_static_ward_closure",
        lambda *args, **kwargs: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(
        material,
        "static_matsubara_kernel_to_sheet_response",
        lambda *args, **kwargs: sheet,
    )
    monkeypatch.setattr(
        material,
        "positive_matsubara_kernel_to_sheet_response",
        lambda *args, **kwargs: pytest.fail("positive path used for xi=0"),
    )

    sample = material.build_material_response_sample(
        result,
        frequency_index=0,
        policy=material.MaterialResponsePolicy(),
    )
    assert sample.frequency_sector == "zero_matsubara"
    assert sample.response is sheet


def test_builder_uses_only_positive_path_for_positive_xi(monkeypatch) -> None:
    q = np.array([0.02, 0.04])
    xi_eV = 0.05
    sheet = _positive_sheet(q, xi_eV)
    validation = _positive_validation()
    kernel = _kernel(q, xi_eV)
    ward = _effective_ward()
    result = SimpleNamespace(
        components=(object(),),
        rhs=(object(),),
        q_model=q,
        xi_eV_values=np.array([xi_eV]),
        material_cache_fingerprint="material-positive",
        operator_ward=_operator_ward(),
        metadata={},
    )

    monkeypatch.setattr(
        material,
        "effective_em_kernel_from_components",
        lambda *args, **kwargs: kernel,
    )
    monkeypatch.setattr(
        material,
        "validate_effective_ward_xy",
        lambda *args, **kwargs: ward,
    )
    monkeypatch.setattr(
        material,
        "positive_matsubara_kernel_to_sheet_response",
        lambda *args, **kwargs: sheet,
    )
    monkeypatch.setattr(
        material,
        "validate_positive_matsubara_sheet_response",
        lambda *args, **kwargs: validation,
    )
    monkeypatch.setattr(
        material,
        "static_matsubara_kernel_to_sheet_response",
        lambda *args, **kwargs: pytest.fail("static path used for xi>0"),
    )

    sample = material.build_material_response_sample(
        result,
        frequency_index=0,
        policy=material.MaterialResponsePolicy(),
    )
    assert sample.frequency_sector == "positive_matsubara"
    assert sample.response is sheet
    assert sample.strict_static_ward is None
