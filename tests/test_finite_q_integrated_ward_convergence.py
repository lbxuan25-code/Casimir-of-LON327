from __future__ import annotations

import numpy as np

from validation.lib.finite_q_integrated_ward_convergence import (
    run_integrated_ward_chain_convergence,
    shifted_uniform_bz_mesh,
)


def test_shifted_uniform_bz_mesh_shape_and_bounds():
    points = shifted_uniform_bz_mesh(3, 0.25, 0.5)

    assert points.shape == (9, 2)
    assert np.all(points >= -np.pi)
    assert np.all(points < np.pi)


def test_integrated_ward_chain_convergence_payload_smoke():
    payload = run_integrated_ward_chain_convergence(
        model_name="symmetry_bdg_2band",
        pairings=("spm",),
        q_values=(0.01,),
        nk_values=(3,),
        shift_fractions=(0.0,),
        omega_eV=0.01,
        delta0_eV=0.1,
    )

    assert payload["identity_version"] == "finite_q_integrated_ward_convergence_v1"
    assert payload["diagnostic_role"] == "integrated_ward_chain_convergence_not_a_new_ward_criterion"
    assert payload["rows"]
    assert payload["summaries"]
    row = payload["rows"][0]
    assert row["chain"]["max_bubble_to_equal_time_difference_norm"] >= 0.0
    assert row["chain"]["max_equal_time_to_contact_difference_norm"] >= 0.0
    assert payload["valid_for_casimir_input"] is False
