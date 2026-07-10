from __future__ import annotations

import numpy as np

from lno327.electrodynamics.conventions import positive_matsubara_kernel_to_sheet_response
from lno327.numerics.grids import uniform_bz_mesh
from lno327.numerics.weights import k_weights
from lno327.response.config import KuboConfig
from lno327.response.effective_kernel import effective_em_kernel_from_components
from lno327.response.finite_q_bdg import finite_q_bdg_response_from_model_ansatz
from validation.lib.finite_q_validation_models import get_finite_q_validation_model


def test_two_band_engine_flows_directly_into_typed_primitive_sheet_contract():
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("spm", phase_vertex="bond_endpoint_gauge")
    pairing_params = model.build_pairing_params(0.1)
    q_model = np.array([0.03, 0.02])
    points = uniform_bz_mesh(2)
    weights = k_weights(points)
    config = KuboConfig.from_kelvin(
        omega_eV=0.02,
        temperature_K=10.0,
        eta_eV=1e-8,
        output_si=False,
    )

    components = finite_q_bdg_response_from_model_ansatz(
        model.spec,
        ansatz,
        config.omega_eV,
        q_model,
        points,
        weights,
        config,
        pairing_params,
    )
    kernel = effective_em_kernel_from_components(
        components,
        q_model=q_model,
        xi_eV=config.omega_eV,
    )
    sheet = positive_matsubara_kernel_to_sheet_response(kernel)

    np.testing.assert_array_equal(kernel.matrix, components.amplitude_phase_schur)
    np.testing.assert_allclose(
        sheet.matrix_model,
        -components.amplitude_phase_schur[1:, 1:] / config.omega_eV,
        rtol=1e-14,
        atol=1e-14,
    )
    assert kernel.metadata["source"] == "BdGFiniteQResponseComponents.amplitude_phase_schur"
    assert sheet.metadata["source"] == "EffectiveEMKernel.spatial_xy"
    assert sheet.metadata["frequency_sector"] == "positive_matsubara"
