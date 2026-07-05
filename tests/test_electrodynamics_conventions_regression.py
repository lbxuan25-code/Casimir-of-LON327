import numpy as np
import pytest

from lno327.electrodynamics.conductivity import ConductivityTensor
import lno327.electrodynamics.conventions as new


def test_sheet_conductivity_convention_objects_and_aliases():
    assert new.ResponseUnitConvention is new.SheetConductivityConvention
    convention = new.SheetConductivityConvention()
    assert convention.apply_e2_over_hbar is True
    assert convention.use_explicit_lattice_geometry is False
    assert convention.lattice_constant_m is None
    assert convention.unit_cell_area_m2 is None
    conversion = new.SheetConductivityConversion(
        tensor=ConductivityTensor(1.0, 2.0),
        unit_stage="model_response",
        unit_label="raw",
        normalization_status="raw",
        valid_for_casimir_input=False,
        notes=("note",),
    )
    assert conversion.unit_stage == "model_response"


def test_sheet_conductivity_conversion_helpers_are_well_formed():
    matrix = np.array([[1.0, 0.2], [0.3, 2.0]], dtype=complex)
    sheet_new = new.model_response_to_sheet_conductivity(matrix)
    assert sheet_new.tensor.matrix().shape == (2, 2)
    assert np.all(np.isfinite(sheet_new.tensor.matrix()))
    assert sheet_new.valid_for_casimir_input is True

    dim_new = new.sheet_conductivity_to_reflection_dimensionless(sheet_new)
    assert dim_new.valid_for_casimir_input is True

    direct_new = new.model_response_to_reflection_dimensionless(matrix)
    np.testing.assert_allclose(direct_new.tensor.matrix(), dim_new.tensor.matrix())

    assert new.require_sheet_conductivity_for_reflection(sheet_new) is sheet_new
    tensor = ConductivityTensor(1.0, 2.0, 0.2, 0.3)
    assert new.sheet_conductivity_to_dimensionless(tensor).matrix().shape == (2, 2)


def test_spatial_response_and_metadata_are_well_formed():
    response = np.arange(9, dtype=float).reshape(3, 3).astype(complex)
    converted = new.spatial_response_to_bilayer_sheet_conductivity_model(response, 0.5)
    assert converted.shape == (2, 2)
    metadata = new.bilayer_sheet_conductivity_convention_metadata()
    assert metadata["not_bulk_3d"] is True
    assert metadata["si_scaling_applied"] is False


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
