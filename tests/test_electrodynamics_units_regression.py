import numpy as np
import pytest

import lno327.electrodynamics.units as new
import lno327.response_conventions as old


def _convention(cls):
    return cls(lattice_a_x_m=3.9e-10, lattice_a_y_m=3.8e-10)


def test_unit_scalar_helpers_match_old_reference():
    for name in (
        "e2_over_hbar_siemens",
        "vacuum_impedance_ohm",
        "four_pi_alpha",
        "z0_e2_over_hbar",
        "dimensionless_sheet_prefactor_square_lattice",
    ):
        np.testing.assert_allclose(getattr(new, name)(), getattr(old, name)(), rtol=1e-12, atol=1e-12)


def test_sheet_conductivity_unit_convention_validation_matches_old_reference():
    for cls in (new.SheetConductivityUnitConvention, old.SheetConductivityUnitConvention):
        with pytest.raises(ValueError, match="lattice_a_x_m must be positive"):
            cls(lattice_a_x_m=0.0, lattice_a_y_m=1.0)
        with pytest.raises(ValueError, match="lattice_a_y_m must be positive"):
            cls(lattice_a_x_m=1.0, lattice_a_y_m=0.0)
        with pytest.raises(ValueError, match="unit_cell_area_m2 must be positive"):
            cls(lattice_a_x_m=1.0, lattice_a_y_m=1.0, unit_cell_area_m2=0.0)


def test_geometry_and_conversion_helpers_match_old_reference():
    new_convention = _convention(new.SheetConductivityUnitConvention)
    old_convention = _convention(old.SheetConductivityUnitConvention)
    sigma_model = np.array([[1.0, 0.2j], [0.3 - 0.1j, 2.0]], dtype=complex)

    for i, j in (("x", "x"), ("x", "y"), ("y", "x"), ("y", "y")):
        np.testing.assert_allclose(
            new.geometry_factor(i, j, new_convention),
            old.geometry_factor(i, j, old_convention),
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        new.sheet_geometry_factor_tensor(new_convention),
        old.sheet_geometry_factor_tensor(old_convention),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        new.model_to_si_sheet_conductivity(sigma_model, new_convention),
        old.model_to_si_sheet_conductivity(sigma_model, old_convention),
        rtol=1e-12,
        atol=1e-12,
    )
    sigma_si = old.model_to_si_sheet_conductivity(sigma_model, old_convention)
    np.testing.assert_allclose(
        new.si_sheet_to_dimensionless_conductivity(sigma_si),
        old.si_sheet_to_dimensionless_conductivity(sigma_si),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        new.model_to_dimensionless_sheet_conductivity(sigma_model, new_convention),
        old.model_to_dimensionless_sheet_conductivity(sigma_model, old_convention),
        rtol=1e-12,
        atol=1e-12,
    )


def test_unit_conversion_metadata_matches_old_reference():
    new_convention = _convention(new.SheetConductivityUnitConvention)
    old_convention = _convention(old.SheetConductivityUnitConvention)
    new_metadata = new.conductivity_unit_conversion_metadata(new_convention)
    old_metadata = old.conductivity_unit_conversion_metadata(old_convention)

    assert new_metadata.keys() == old_metadata.keys()
    for key in new_metadata:
        if isinstance(new_metadata[key], np.ndarray):
            np.testing.assert_allclose(new_metadata[key], old_metadata[key], rtol=1e-12, atol=1e-12)
        else:
            assert new_metadata[key] == old_metadata[key]
