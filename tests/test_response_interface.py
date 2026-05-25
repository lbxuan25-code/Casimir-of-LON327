from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from lno327 import (
    LocalSheetResponse,
    PairingAmplitudes,
    conductivity_tensor_from_matrix,
    k_weights,
    local_response_imag_axis,
    uniform_bz_mesh,
    validate_local_response_symmetry,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_local_sheet_response_imag.py"
SPEC = spec_from_file_location("compare_local_sheet_response_imag", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_script = module_from_spec(SPEC)
SPEC.loader.exec_module(compare_script)


def test_local_response_imag_axis_normal_returns_local_sheet_response():
    mesh = uniform_bz_mesh(3)
    weights = k_weights(mesh)

    response = local_response_imag_axis(
        "normal",
        omega_eV=0.1,
        k_points=mesh,
        temperature_K=30.0,
        eta_eV=0.02,
        k_weights=weights,
    )

    assert isinstance(response, LocalSheetResponse)
    assert response.kind == "normal"
    assert response.matrix.shape == (2, 2)
    assert not response.valid_for_casimir_input


def test_local_response_imag_axis_bdg_returns_local_sheet_response_for_positive_omega():
    mesh = uniform_bz_mesh(3)
    weights = k_weights(mesh)

    for kind in ("spm", "dwave"):
        response = local_response_imag_axis(
            kind,
            omega_eV=0.1,
            k_points=mesh,
            temperature_K=30.0,
            eta_eV=0.02,
            pairing_params=PairingAmplitudes(delta0_eV=0.04),
            k_weights=weights,
        )

        assert isinstance(response, LocalSheetResponse)
        assert response.kind == kind
        assert response.matrix.shape == (2, 2)
        assert not response.valid_for_casimir_input


def test_local_response_imag_axis_bdg_rejects_zero_omega():
    mesh = uniform_bz_mesh(3)

    with pytest.raises(ValueError, match="n=0 is unresolved"):
        local_response_imag_axis(
            "spm",
            omega_eV=0.0,
            k_points=mesh,
            temperature_K=30.0,
            pairing_params=PairingAmplitudes(delta0_eV=0.04),
        )


def test_conductivity_tensor_from_matrix_maps_components():
    matrix = np.array([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]])

    tensor = conductivity_tensor_from_matrix(matrix)

    assert tensor.xx == matrix[0, 0]
    assert tensor.yy == matrix[1, 1]
    assert tensor.xy == matrix[0, 1]
    assert tensor.yx == matrix[1, 0]


def test_validate_local_response_symmetry_passes_isotropic_matrix():
    response = LocalSheetResponse(
        kind="normal",
        omega_eV=0.1,
        matrix=np.array([[2.0 + 0.0j, 0.0], [0.0, 2.0 + 0.0j]], dtype=complex),
        unit_label="test",
        source="test",
        valid_for_casimir_input=False,
        notes=(),
    )

    diagnostics = validate_local_response_symmetry(response)

    assert diagnostics["delta"] == 0.0
    assert diagnostics["relative_offdiag"] == 0.0
    assert diagnostics["relative_eigen_split"] == 0.0
    assert diagnostics["isotropic_within_tolerance"]


def test_compare_local_sheet_response_script_returns_nonempty_data(tmp_path):
    data = compare_script.scan_responses(
        kinds=["normal", "spm", "dwave"],
        delta0_eV=0.04,
        nk=4,
        temperature_K=30.0,
        matsubara_min=1,
        matsubara_max=1,
        eta_eV=1e-4,
    )
    npz_path, *_ = compare_script.save_outputs(data, tmp_path / "local_sheet_response")

    assert data["kind"].size == 3
    assert set(["normal", "spm", "dwave"]) == set(data["kind"])
    assert set(compare_script.REQUIRED_NPZ_FIELDS).issubset(data)
    with np.load(npz_path, allow_pickle=True) as loaded:
        assert set(compare_script.REQUIRED_NPZ_FIELDS).issubset(loaded.files)
