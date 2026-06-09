# Normal-state Pi_mu_nu Ward identity prototype

This is a normal-state Pi_mu_nu Ward diagnostic.
It compares midpoint velocity and Peierls current vertex schemes.
It is not conductivity and not a reflection/Casimir input.
The contact term is not included.
Large Ward residuals may reflect finite-q vertex/contact-term closure gaps, not a material conclusion.

run_command = `python validation/scripts/response/diagnose_normal_ward_identity.py --quick`
quick_mode=True
expanded_data_written=False
density_current_included=True
current_current_included=True
diamagnetic_contact_included=False
contact_scheme=none
normal_state_only=True
bdg_computed=False
conductivity_computed=False
casimir_computed=False
not_final_casimir_conclusion=True
not_final_finite_q_conductivity=True

## Parameter grid
- vertex_schemes = midpoint peierls
- matsubara_n_list = 1
- temperature_K = 30
- q_list = 0.001 0.01 0.1
- q_angle_list = 0 0.785398
- nk_list = 8
- degeneracy_tol_eV = 1e-10

## Ward residual summary by vertex scheme
- midpoint: max left Ward error = 0.070597
- midpoint: max right Ward error = 0.070597
- midpoint: max Ward error = 0.070597
- peierls: max left Ward error = 0.0705549
- peierls: max right Ward error = 0.0705549
- peierls: max Ward error = 0.0705549

## q_model max-error trend
- midpoint: q=0.001:0.00070669, q=0.01:0.00706672, q=0.1:0.070597
- peierls: q=0.001:0.00070669, q=0.01:0.00706668, q=0.1:0.0705549

## Comparison
- Peierls current vertex does not materially lower the max Ward residual in this prototype (midpoint/Peierls factor = 1.0006); possible reasons include the missing contact term or remaining vertex/contact closure gaps.

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/data/normal_ward_identity_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/left_right_ward_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q_by_vertex_scheme.png
