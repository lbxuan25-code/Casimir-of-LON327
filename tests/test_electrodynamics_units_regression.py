import numpy as np
import pytest

import lno327.electrodynamics.units as new


def _convention(cls):
    return cls(lattice_a_x_m=3.9e-10, lattice_a_y_m=3.8e-10)


def test_unit_scalar_helpers_are_positive():
    for name in (
        "e2_over_hbar_siemens",
        "vacuum_impedance_ohm",
        "four_pi_alpha",
        "z0_e2_over_hbar",
        "dimensionless_sheet_prefactor_square_lattice",
    ):
        assert getattr(new, name)() > 0.0


def test_sheet_conductivity_unit_convention_validation():
    with pytest.raises(ValueError, match="lattice_a_x_m must be positive"):
        new.SheetConductivityUnitConvention(lattice_a_x_m=0.0, lattice_a_y_m=1.0)
    with pytest.raises(ValueError, match="lattice_a_y_m must be positive"):
        new.SheetConductivityUnitConvention(lattice_a_x_m=1.0, lattice_a_y_m=0.0)
    with pytest.raises(ValueError, match="unit_cell_area_m2 must be positive"):
        new.SheetConductivityUnitConvention(lattice_a_x_m=1.0, lattice_a_y_m=1.0, unit_cell_area_m2=0.0)


def test_geometry_and_conversion_helpers_are_well_formed():
    new_convention = _convention(new.SheetConductivityUnitConvention)
    sigma_model = np.array([[1.0, 0.2j], [0.3 - 0.1j, 2.0]], dtype=complex)

    for i, j in (("x", "x"), ("x", "y"), ("y", "x"), ("y", "y")):
        assert new.geometry_factor(i, j, new_convention) > 0.0
    assert new.sheet_geometry_factor_tensor(new_convention).shape == (2, 2)
    sigma_si = new.model_to_si_sheet_conductivity(sigma_model, new_convention)
    sigma_tilde = new.si_sheet_to_dimensionless_conductivity(sigma_si)
    sigma_direct = new.model_to_dimensionless_sheet_conductivity(sigma_model, new_convention)
    assert sigma_si.shape == (2, 2)
    assert sigma_tilde.shape == (2, 2)
    np.testing.assert_allclose(sigma_direct, sigma_tilde)


def test_unit_conversion_metadata_is_well_formed():
    new_convention = _convention(new.SheetConductivityUnitConvention)
    new_metadata = new.conductivity_unit_conversion_metadata(new_convention)

    assert "reflection_casimir_ready" in new_metadata
    assert new_metadata["reflection_casimir_ready"] is False
