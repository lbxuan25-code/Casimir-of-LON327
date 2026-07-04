import numpy as np
import pytest

from lno327.electrodynamics.conductivity import ConductivityTensor
import lno327.electrodynamics.conventions as new
import lno327.response_conventions as old


def _assert_conversion_matches(actual, expected):
    np.testing.assert_allclose(actual.tensor.matrix(), expected.tensor.matrix(), rtol=1e-12, atol=1e-12)
    assert actual.unit_stage == expected.unit_stage
    assert actual.unit_label == expected.unit_label
    assert actual.normalization_status == expected.normalization_status
    assert actual.valid_for_casimir_input == expected.valid_for_casimir_input
    assert actual.notes == expected.notes


def test_sheet_conductivity_convention_objects_and_aliases():
    assert new.ResponseUnitConvention is new.SheetConductivityConvention
    convention = new.SheetConductivityConvention()
    old_convention = old.SheetConductivityConvention()
    assert convention.apply_e2_over_hbar == old_convention.apply_e2_over_hbar
    assert convention.use_explicit_lattice_geometry == old_convention.use_explicit_lattice_geometry
    assert convention.lattice_constant_m == old_convention.lattice_constant_m
    assert convention.unit_cell_area_m2 == old_convention.unit_cell_area_m2
    conversion = new.SheetConductivityConversion(
        tensor=ConductivityTensor(1.0, 2.0),
        unit_stage="model_response",
        unit_label="raw",
        normalization_status="raw",
        valid_for_casimir_input=False,
        notes=("note",),
    )
    assert conversion.unit_stage == "model_response"


def test_sheet_conductivity_conversion_helpers_match_old_reference():
    matrix = np.array([[1.0, 0.2], [0.3, 2.0]], dtype=complex)
    sheet_new = new.model_response_to_sheet_conductivity(matrix)
    sheet_old = old.model_response_to_sheet_conductivity(matrix)
    _assert_conversion_matches(sheet_new, sheet_old)

    dim_new = new.sheet_conductivity_to_reflection_dimensionless(sheet_new)
    dim_old = old.sheet_conductivity_to_reflection_dimensionless(sheet_old)
    _assert_conversion_matches(dim_new, dim_old)

    direct_new = new.model_response_to_reflection_dimensionless(matrix)
    direct_old = old.model_response_to_reflection_dimensionless(matrix)
    _assert_conversion_matches(direct_new, direct_old)

    assert new.require_sheet_conductivity_for_reflection(sheet_new) is sheet_new
    tensor = ConductivityTensor(1.0, 2.0, 0.2, 0.3)
    old_tensor = old.ConductivityTensor(1.0, 2.0, 0.2, 0.3)
    np.testing.assert_allclose(
        new.sheet_conductivity_to_dimensionless(tensor).matrix(),
        old.sheet_conductivity_to_dimensionless(old_tensor).matrix(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_spatial_response_and_metadata_match_old_reference():
    response = np.arange(9, dtype=float).reshape(3, 3).astype(complex)
    np.testing.assert_allclose(
        new.spatial_response_to_bilayer_sheet_conductivity_model(response, 0.5),
        old.spatial_response_to_bilayer_sheet_conductivity_model(response, 0.5),
        rtol=1e-12,
        atol=1e-12,
    )
    assert new.bilayer_sheet_conductivity_convention_metadata() == old.bilayer_sheet_conductivity_convention_metadata()


def test_convention_error_behavior_matches_old_reference():
    matrix = np.eye(2, dtype=complex)
    sheet = new.model_response_to_sheet_conductivity(matrix)
    with pytest.raises(ValueError, match="twice"):
        new.model_response_to_sheet_conductivity(sheet)
    with pytest.raises(ValueError, match="shape"):
        new.model_response_to_sheet_conductivity(np.eye(3))
    with pytest.raises(ValueError, match="apply_e2_over_hbar"):
        new.model_response_to_sheet_conductivity(matrix, new.SheetConductivityConvention(apply_e2_over_hbar=False))
    with pytest.raises(ValueError, match="explicit lattice geometry"):
        new.model_response_to_sheet_conductivity(
            matrix,
            new.SheetConductivityConvention(use_explicit_lattice_geometry=True),
        )
    with pytest.raises(ValueError, match="omega_eV must be positive"):
        new.spatial_response_to_bilayer_sheet_conductivity_model(np.eye(3), 0.0)
