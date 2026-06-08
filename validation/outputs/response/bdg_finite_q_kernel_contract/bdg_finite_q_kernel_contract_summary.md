# BdG finite-q current-current kernel contract diagnostic

This is BdG finite-q current-current kernel contract only.
It is not gauge-closed finite-q conductivity.
It is not Pi_mu_nu and not Ward identity.
It is not Casimir input.
Stage 2 showed Casimir-relevant q_model reaches O(1), so this diagnostic includes q up to O(1) but remains a kernel contract test.

run_command = `python validation/scripts/response/diagnose_bdg_finite_q_kernel.py`
quick_mode=False
expanded_data_written=False
kernel_block_only=True
current_current_only=True
positive_matsubara_only=True
response_computed=True
conductivity_computed=False
pi_mu_nu_computed=False
ward_identity_checked=False
casimir_computed=False
not_final_casimir_conclusion=True

## Parameter grid
- kinds = spm dwave
- delta0_list = 0 1e-05 0.0001 0.001 0.01 0.04
- matsubara_n_list = 1 2 4
- temperature_K = 30
- q_list = 0 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.5 1
- q_angle_list = 0 0.392699 0.785398 1.1781 1.5708
- nk_list = 8 12 16
- degeneracy_tol_eV = 1e-10
- q_model_max = 1

## Contract results
- max q->0 same-interface error for 0<q<=1e-3: 0.000276563
- max Delta0=0 normal finite-q kernel comparison error: 5.84509e-15
- max C4 covariance error: 1.20029e-14

The Delta0=0 comparison is against normal_current_current_kernel_imag_axis and is reported as a kernel comparison, not conductivity.
No legacy local BdG comparison is used in the pass/fail path.

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/bdg_finite_q_kernel_contract/data/bdg_finite_q_kernel_contract_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/bdg_finite_q_kernel_contract/figures/q_to_zero_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/bdg_finite_q_kernel_contract/figures/normal_limit_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/bdg_finite_q_kernel_contract/figures/c4_covariance_error_vs_q.png
