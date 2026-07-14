from __future__ import annotations

import numpy as np
import pytest

from lno327.constants import KB_EV_PER_K
from validation.lib.finite_q_validation_models import get_finite_q_validation_model
from validation.lib.positive_orbit_gauss import integrate_positive_orbit_gauss


@pytest.mark.parametrize(
    ("pairing_name", "expected_policy"),
    [
        ("spm", "q_independent"),
        ("dwave", "nearest_neighbor_bond_metric"),
    ],
)
def test_positive_orbit_gauss_uses_one_batched_backend_with_pairing_policy(
    pairing_name: str,
    expected_policy: str,
) -> None:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz(pairing_name, phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    temperature = 10.0
    xi = np.asarray([2.0 * np.pi * KB_EV_PER_K * temperature])

    result = integrate_positive_orbit_gauss(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=temperature,
        eta_eV=1e-8,
        nk=4,
        mx=1,
        my=0,
        transverse_order=4,
        panel_count=1,
        max_point_evaluations=64,
        transverse_workers=1,
        transverse_task_size=2,
    )

    assert result.pairing_name == pairing_name
    assert result.phase_hessian_policy == expected_policy
    assert result.evaluator_profile.callbacks == 4
    assert result.evaluator_profile.complete_orbit_points == 32
    assert result.evaluator_profile.material_workspace_implementation == (
        "batched_model_capability"
    )
    assert result.evaluator_profile.q_workspace_implementation == (
        "batched_model_capability"
    )
    assert result.quadrature.full_transverse_period_integrated is True
    assert result.quadrature.symmetry_reduction_applied is False
    assert result.components[0].metadata["pairing"] == pairing_name
    assert result.components[0].metadata["post_integral_phase_hessian_policy"] == (
        expected_policy
    )


def test_spm_fork_process_gauss_matches_serial() -> None:
    model = get_finite_q_validation_model("symmetry_bdg_2band")
    ansatz = model.build_ansatz("spm", phase_vertex="bond_endpoint_gauge")
    pairing = model.build_pairing_params(0.1)
    temperature = 10.0
    xi = np.asarray([2.0 * np.pi * KB_EV_PER_K * temperature])

    common = dict(
        spec=model.spec,
        ansatz=ansatz,
        pairing=pairing,
        xi_eV_values=xi,
        temperature_K=temperature,
        eta_eV=1e-8,
        nk=4,
        mx=1,
        my=0,
        transverse_order=4,
        panel_count=1,
        max_point_evaluations=64,
        transverse_task_size=2,
    )
    serial = integrate_positive_orbit_gauss(**common, transverse_workers=1)
    parallel = integrate_positive_orbit_gauss(**common, transverse_workers=2)

    np.testing.assert_array_equal(parallel.quadrature.value, serial.quadrature.value)
    assert parallel.quadrature.execution_strategy == (
        "fork_process_transverse_nodes_ordered_parent_reduction"
    )
