# Response-level Ward convention verification

This is Stage 4.4 response-level Ward convention verification.
It compares Hamiltonian-vertex and physical-current convention cases for normal-state Pi_mu_nu.
It is not final finite-q conductivity, not reflection/Casimir input, and not a material conclusion.
The best residual case must not be promoted to final physics until it is consistent with the analytic convention.

run_command = `python validation/scripts/response/verify_response_level_ward_conventions.py`
quick_mode=False
expanded_data_written=False
response_computed=True
conductivity_computed=False
casimir_computed=False
normal_state_only=True
not_final_finite_q_conductivity=True
not_final_casimir_conclusion=True

## Convention cases
- current_code_phys_q_plus: current_vertex_multiplier=1.0, contact_sign_convention=plus, ward_q_sign=1.0. Current diagnostic baseline: code-like plus-q residual, not a final convention claim.
- hamiltonian_vertex_q_minus_contact_plus: current_vertex_multiplier=1.0, contact_sign_convention=plus, ward_q_sign=-1.0. Hamiltonian derivative vertex convention: Gamma_i^H with Q_H=(iOmega,-qx,-qy).
- physical_current_q_plus_contact_minus: current_vertex_multiplier=-1.0, contact_sign_convention=minus, ward_q_sign=1.0. Physical-current convention: -Gamma_i^H with Q_phys=(iOmega,+qx,+qy).
- physical_current_q_plus_contact_plus: current_vertex_multiplier=-1.0, contact_sign_convention=plus, ward_q_sign=1.0. Physical-current control with contact plus.
- hamiltonian_vertex_q_minus_contact_minus: current_vertex_multiplier=1.0, contact_sign_convention=minus, ward_q_sign=-1.0. Hamiltonian-vertex control with contact minus.

All cases use vertex_scheme=peierls and contact_scheme=finite_q_peierls.
The script constructs contact-only response by subtracting contact_scheme=none from finite_q_peierls plus.

## Parameter grid
- matsubara_n_list = 1 2 4
- temperature_K = 30
- q_list = 0.001 0.005 0.01 0.05 0.1
- q_angle_list = 0 0.785398 1.5708
- nk_list = 8 12 16
- degeneracy_tol_eV = 1e-10

## Max residuals by case
- current_code_phys_q_plus: max full residual_abs = 0.122067
- current_code_phys_q_plus: max density residual_abs = 0.0741126
- current_code_phys_q_plus: max spatial residual_abs = 0.122067
- hamiltonian_vertex_q_minus_contact_plus: max full residual_abs = 0.127494
- hamiltonian_vertex_q_minus_contact_plus: max density residual_abs = 1.18341e-15
- hamiltonian_vertex_q_minus_contact_plus: max spatial residual_abs = 0.127494
- physical_current_q_plus_contact_minus: max full residual_abs = 0.0414927
- physical_current_q_plus_contact_minus: max density residual_abs = 1.18341e-15
- physical_current_q_plus_contact_minus: max spatial residual_abs = 0.0414927
- physical_current_q_plus_contact_plus: max full residual_abs = 0.127494
- physical_current_q_plus_contact_plus: max density residual_abs = 1.18341e-15
- physical_current_q_plus_contact_plus: max spatial residual_abs = 0.127494
- hamiltonian_vertex_q_minus_contact_minus: max full residual_abs = 0.0414927
- hamiltonian_vertex_q_minus_contact_minus: max density residual_abs = 1.18341e-15
- hamiltonian_vertex_q_minus_contact_minus: max spatial residual_abs = 0.0414927

## Small-q alpha from q_model = 0.001, 0.005, 0.01
- current_code_phys_q_plus: full alpha = 0.999996
- current_code_phys_q_plus: density alpha = 1.99883
- current_code_phys_q_plus: spatial alpha = 0.999996
- hamiltonian_vertex_q_minus_contact_plus: full alpha = 1.00003
- hamiltonian_vertex_q_minus_contact_plus: density alpha = -0.124694
- hamiltonian_vertex_q_minus_contact_plus: spatial alpha = 1.00003
- physical_current_q_plus_contact_minus: full alpha = 0.999993
- physical_current_q_plus_contact_minus: density alpha = -0.124694
- physical_current_q_plus_contact_minus: spatial alpha = 0.999993
- physical_current_q_plus_contact_plus: full alpha = 1.00003
- physical_current_q_plus_contact_plus: density alpha = -0.124694
- physical_current_q_plus_contact_plus: spatial alpha = 1.00003
- hamiltonian_vertex_q_minus_contact_minus: full alpha = 0.999993
- hamiltonian_vertex_q_minus_contact_minus: density alpha = -0.124694
- hamiltonian_vertex_q_minus_contact_minus: spatial alpha = 0.999993

## q_model max residual trend
- current_code_phys_q_plus spatial: q=0.001:0.000764092, q=0.005:0.00382045, q=0.01:0.00764083, q=0.05:0.0555555, q=0.1:0.122067
- current_code_phys_q_plus full: q=0.001:0.000764092, q=0.005:0.00382045, q=0.01:0.00764083, q=0.05:0.0555555, q=0.1:0.122067
- hamiltonian_vertex_q_minus_contact_plus spatial: q=0.001:0.000794706, q=0.005:0.00397359, q=0.01:0.00794759, q=0.05:0.0606769, q=0.1:0.127494
- hamiltonian_vertex_q_minus_contact_plus full: q=0.001:0.000794706, q=0.005:0.00397359, q=0.01:0.00794759, q=0.05:0.0606769, q=0.1:0.127494
- physical_current_q_plus_contact_minus spatial: q=0.001:0.000416418, q=0.005:0.00208208, q=0.01:0.0041641, q=0.05:0.0208111, q=0.1:0.0414927
- physical_current_q_plus_contact_minus full: q=0.001:0.000416418, q=0.005:0.00208208, q=0.01:0.0041641, q=0.05:0.0208111, q=0.1:0.0414927
- physical_current_q_plus_contact_plus spatial: q=0.001:0.000794706, q=0.005:0.00397359, q=0.01:0.00794759, q=0.05:0.0606769, q=0.1:0.127494
- physical_current_q_plus_contact_plus full: q=0.001:0.000794706, q=0.005:0.00397359, q=0.01:0.00794759, q=0.05:0.0606769, q=0.1:0.127494
- hamiltonian_vertex_q_minus_contact_minus spatial: q=0.001:0.000416418, q=0.005:0.00208208, q=0.01:0.0041641, q=0.05:0.0208111, q=0.1:0.0414927
- hamiltonian_vertex_q_minus_contact_minus full: q=0.001:0.000416418, q=0.005:0.00208208, q=0.01:0.0041641, q=0.05:0.0208111, q=0.1:0.0414927

## Interpretation
- Best case for max spatial residual_abs: physical_current_q_plus_contact_minus.
- Best case for spatial small-q alpha: hamiltonian_vertex_q_minus_contact_plus with alpha = 1.00003.
- No case raises the spatial alpha close to 2 on this grid; the spatial residual remains effectively O(q)-like.
- Physical-current convention with contact minus is the best spatial-residual case in this diagnostic; physical current sign and contact minus should be treated as the next implementation candidate, subject to the analytic convention.
- If both Hamiltonian and physical-current conventions remain unclosed, the next checks should include Kubo bubble sign, equal-time/commutator term, density vertex sign, denominator convention, and response index order.

## Output files
- compact_csv = `validation/outputs/response/response_level_ward_conventions/data/response_level_ward_conventions_compact.csv`
- expanded_csv = `validation/outputs/response/response_level_ward_conventions/data/response_level_ward_conventions_expanded.csv`
- expanded_npz = `validation/outputs/response/response_level_ward_conventions/data/response_level_ward_conventions_expanded.npz`
Expanded CSV/NPZ are written only when --write-expanded-data is passed.

## Figures
- `validation/outputs/response/response_level_ward_conventions/figures/spatial_residual_vs_q_by_convention.png`
- `validation/outputs/response/response_level_ward_conventions/figures/full_residual_vs_q_by_convention.png`
- `validation/outputs/response/response_level_ward_conventions/figures/spatial_alpha_by_convention.png`
- `validation/outputs/response/response_level_ward_conventions/figures/density_alpha_by_convention.png`

## Explicit boundary
This stage is convention verification only.
It is not final finite-q conductivity.
It is not reflection/Casimir input.
It is not a material conclusion.
Do not directly treat the best residual case as the final physical implementation without analytic convention closure.
