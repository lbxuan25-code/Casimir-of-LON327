# Normal-state Pi_mu_nu Ward identity prototype

This is a normal-state Pi_mu_nu Ward prototype.
It is not conductivity and not a reflection/Casimir input.
Current vertices currently use midpoint velocity.
The diamagnetic/contact term is not included.
Large Ward residuals may reflect finite-q vertex/contact-term closure gaps, not a material conclusion.

run_command = `python validation/scripts/response/diagnose_normal_ward_identity.py`
quick_mode=False
expanded_data_written=False
density_current_included=True
current_current_included=True
diamagnetic_contact_included=False
normal_state_only=True
bdg_computed=False
conductivity_computed=False
casimir_computed=False
not_final_casimir_conclusion=True

## Parameter grid
- matsubara_n_list = 1 2 4
- temperature_K = 30
- q_list = 0.001 0.005 0.01 0.05 0.1
- q_angle_list = 0 0.785398 1.5708
- nk_list = 8 12 16
- degeneracy_tol_eV = 1e-10

## Ward residual summary
- max left Ward error = 0.0801347
- max right Ward error = 0.0801347
- max Ward error = 0.0801347

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/data/normal_ward_identity_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/left_right_ward_error_vs_q.png
