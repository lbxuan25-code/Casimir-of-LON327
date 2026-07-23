"""Numerical equivalence tests for geometry assembly from material responses."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lno327.casimir.lifshitz_integrand import passive_sheet_logdet
from lno327.casimir.material_response import MaterialResponseSample
from lno327.casimir.material_two_plate import (
    TwoPlateGeometryPolicy,
    assemble_two_plate_logdet,
)
from lno327.electrodynamics.conductivity import ConductivityTensor
from lno327.electrodynamics.conventions import (
    PositiveMatsubaraSheetResponse,
    SheetConductivityConversion,
    SheetResponseValidation,
)
from lno327.electrodynamics.reflection import (
    positive_matsubara_sheet_response_to_reflection,
)
from lno327.electrodynamics.static_sheet import (
    STATIC_LOCAL_BASIS,
    StaticSheetResponse,
    StaticSheetValidation,
    static_sheet_response_to_reflection,
)


def _ward() -> SimpleNamespace:
    side = SimpleNamespace(effective_mixed_ratio=0.0)
    return SimpleNamespace(
        passed=True,
        left=side,
        right=side,
        schur_condition_number=1.0,
    )


def _static_sample(q: np.ndarray, *, chi: float, stiffness: float) -> MaterialResponseSample:
    validation = StaticSheetValidation(
        finite=True,
        ward_passed=True,
        relative_imaginary_norm=0.0,
        relative_longitudinal_gauge_residual=0.0,
        relative_density_transverse_mixing=0.0,
        chi_bar=chi,
        dbar_t=stiffness,
        reality_tolerance=1e-9,
        longitudinal_tolerance=1e-7,
        mixing_tolerance=1e-7,
        passivity_tolerance=1e-10,
    )
    response = StaticSheetResponse(
        kernel_lt=np.diag([-chi, 0.0, -stiffness]).astype(complex),
        chi_bar=chi,
        dbar_t=stiffness,
        q_model=q,
        energy_scale_eV=1.0,
        degeneracy=1.0,
        basis=STATIC_LOCAL_BASIS,
        validation=validation,
        metadata={"source": "test"},
    )
    return MaterialResponseSample(
        frequency_index=0,
        frequency_sector="zero_matsubara",
        q_crystal=q,
        xi_eV=0.0,
        material_cache_fingerprint="static-test",
        kernel=SimpleNamespace(q_model=q, xi_eV=0.0),
        operator_ward=SimpleNamespace(passed=True),
        effective_ward=_ward(),
        strict_static_ward=SimpleNamespace(passed=True),
        response=response,
        sheet_validation=validation,
        metadata={},
    )


def _positive_sample(q: np.ndarray, *, xi_eV: float, scale: float) -> MaterialResponseSample:
    tensor = ConductivityTensor(scale, 1.4 * scale, 0.0, 0.0)
    sheet = SheetConductivityConversion(
        tensor=tensor,
        unit_stage="sheet_conductivity",
        unit_label="test-sheet",
        normalization_status="test",
        valid_for_casimir_input=True,
        notes=(),
    )
    tilde = SheetConductivityConversion(
        tensor=tensor,
        unit_stage="reflection_dimensionless_conductivity",
        unit_label="test-tilde",
        normalization_status="test",
        valid_for_casimir_input=True,
        notes=(),
    )
    response = PositiveMatsubaraSheetResponse(
        sigma_model_xy=tensor,
        sigma_sheet_si_xy=sheet,
        sigma_tilde_xy=tilde,
        q_model=q,
        xi_eV=xi_eV,
        degeneracy=1.0,
        basis="crystal_xy",
        metadata={"source": "test"},
    )
    validation = SheetResponseValidation(
        finite=True,
        relative_imaginary_norm=0.0,
        relative_symmetry_residual=0.0,
        minimum_symmetric_eigenvalue=scale,
        reality_tolerance=1e-9,
        symmetry_tolerance=1e-9,
        passivity_tolerance=1e-10,
    )
    return MaterialResponseSample(
        frequency_index=0,
        frequency_sector="positive_matsubara",
        q_crystal=q,
        xi_eV=xi_eV,
        material_cache_fingerprint="positive-test",
        kernel=SimpleNamespace(q_model=q, xi_eV=xi_eV),
        operator_ward=SimpleNamespace(passed=True),
        effective_ward=_ward(),
        strict_static_ward=None,
        response=response,
        sheet_validation=validation,
        metadata={},
    )


def test_two_plate_module_has_no_microscopic_fallback_imports() -> None:
    path = Path("src/lno327/casimir/material_two_plate.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = (
        "lno327.workflows",
        "lno327.response.arbitrary_q",
        "lno327.casimir.material_response_engine",
        "lno327.casimir.material_response_certification",
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


def test_static_assembly_matches_direct_legacy_formula_path() -> None:
    q_lab = np.array([0.02, -0.015])
    sample_1 = _static_sample(q_lab, chi=0.5, stiffness=0.25)
    sample_2 = _static_sample(q_lab, chi=0.6, stiffness=0.3)
    separation_m = 20e-9

    direct_1 = static_sheet_response_to_reflection(
        sample_1.response,
        q_lab_model=q_lab,
        theta_rad=0.0,
    )
    direct_2 = static_sheet_response_to_reflection(
        sample_2.response,
        q_lab_model=q_lab,
        theta_rad=0.0,
    )
    direct_point = passive_sheet_logdet(
        direct_1,
        direct_2,
        separation_m=separation_m,
    )

    assembled = assemble_two_plate_logdet(
        sample_1,
        sample_2,
        q_lab=q_lab,
        theta_1_rad=0.0,
        theta_2_rad=0.0,
        policy=TwoPlateGeometryPolicy(separation_m=separation_m),
    )
    assert assembled.logdet == direct_point.logdet
    assert np.array_equal(assembled.point.trace_log_matrix, direct_point.trace_log_matrix)
    assert assembled.metadata["microscopic_integration_performed"] is False
    assert assembled.metadata["geometry_assembly_only"] is True
    assert assembled.metadata["production_casimir_allowed"] is False


def test_positive_assembly_matches_direct_formula_path() -> None:
    q_lab = np.array([0.01, 0.025])
    xi_eV = 0.05
    sample_1 = _positive_sample(q_lab, xi_eV=xi_eV, scale=0.2)
    sample_2 = _positive_sample(q_lab, xi_eV=xi_eV, scale=0.25)
    separation_m = 30e-9

    direct_1 = positive_matsubara_sheet_response_to_reflection(
        sample_1.response,
        q_lab_model=q_lab,
        theta_rad=0.0,
        lattice_constant_m=3.7544e-10,
    )
    direct_2 = positive_matsubara_sheet_response_to_reflection(
        sample_2.response,
        q_lab_model=q_lab,
        theta_rad=0.0,
        lattice_constant_m=3.7544e-10,
    )
    direct_point = passive_sheet_logdet(
        direct_1,
        direct_2,
        separation_m=separation_m,
    )

    policy = TwoPlateGeometryPolicy(separation_m=separation_m)
    assembled = assemble_two_plate_logdet(
        sample_1,
        sample_2,
        q_lab=q_lab,
        theta_1_rad=0.0,
        theta_2_rad=0.0,
        policy=policy,
    )
    assert np.isclose(assembled.logdet, direct_point.logdet, rtol=0.0, atol=0.0)
    assert np.array_equal(assembled.point.product_eigenvalues, direct_point.product_eigenvalues)
