# Normal-state Pi_mu_nu Ward identity prototype

This is a normal-state Pi_mu_nu Ward diagnostic.
It compares midpoint velocity, Peierls current vertex, and q=0 mass contact diagnostic schemes.
It is not conductivity and not a reflection/Casimir input.
The q0_mass_diagnostic contact is a q=0 mass small-q diagnostic only, not the final finite-q contact.
Large Ward residuals may reflect finite-q vertex/contact-term closure gaps, not a material conclusion.

run_command = `python validation/scripts/response/diagnose_normal_ward_identity.py`
quick_mode=False
expanded_data_written=False
density_current_included=True
current_current_included=True
diamagnetic_contact_included=True only for contact_scheme=q0_mass_diagnostic
contact_scheme=none or q0_mass_diagnostic
not_final_finite_q_contact=True
normal_state_only=True
bdg_computed=False
conductivity_computed=False
casimir_computed=False
not_final_casimir_conclusion=True
not_final_finite_q_conductivity=True

## Parameter grid
- vertex_schemes = midpoint peierls
- contact_schemes = none q0_mass_diagnostic
- contact_sign_conventions = plus minus
- matsubara_n_list = 1 2 4
- temperature_K = 30
- q_list = 0.001 0.005 0.01 0.05 0.1
- q_angle_list = 0 0.785398 1.5708
- nk_list = 8 12 16
- degeneracy_tol_eV = 1e-10

## Ward residual summary by vertex/contact scheme
- midpoint + none: max left Ward error = 0.0801347
- midpoint + none: max right Ward error = 0.0801347
- midpoint + none: max Ward error = 0.0801347
- midpoint + none: max Ward error for q_model <= 0.01 = 0.00706672
- peierls + none: max left Ward error = 0.0801344
- peierls + none: max right Ward error = 0.0801344
- peierls + none: max Ward error = 0.0801344
- peierls + none: max Ward error for q_model <= 0.01 = 0.00706668
- peierls + q0_mass_diagnostic + plus: max left Ward error = 0.0734346
- peierls + q0_mass_diagnostic + plus: max right Ward error = 0.0734346
- peierls + q0_mass_diagnostic + plus: max Ward error = 0.0734346
- peierls + q0_mass_diagnostic + plus: max Ward error for q_model <= 0.01 = 0.00707007
- peierls + q0_mass_diagnostic + minus: max left Ward error = 0.100674
- peierls + q0_mass_diagnostic + minus: max right Ward error = 0.100674
- peierls + q0_mass_diagnostic + minus: max Ward error = 0.100674
- peierls + q0_mass_diagnostic + minus: max Ward error for q_model <= 0.01 = 0.00772246
- midpoint + q0_mass_diagnostic + minus: max left Ward error = 0.100668
- midpoint + q0_mass_diagnostic + minus: max right Ward error = 0.100668
- midpoint + q0_mass_diagnostic + minus: max Ward error = 0.100668
- midpoint + q0_mass_diagnostic + minus: max Ward error for q_model <= 0.01 = 0.00772246
- midpoint + q0_mass_diagnostic + plus: max left Ward error = 0.0734551
- midpoint + q0_mass_diagnostic + plus: max right Ward error = 0.0734551
- midpoint + q0_mass_diagnostic + plus: max Ward error = 0.0734551
- midpoint + q0_mass_diagnostic + plus: max Ward error for q_model <= 0.01 = 0.00707008

## q_model max-error trend
- midpoint + none: q=0.001:0.00070669, q=0.005:0.00353343, q=0.01:0.00706672, q=0.05:0.0418758, q=0.1:0.0801347
- peierls + none: q=0.001:0.00070669, q=0.005:0.00353342, q=0.01:0.00706668, q=0.05:0.0418803, q=0.1:0.0801344
- peierls + q0_mass_diagnostic + plus: q=0.001:0.000707012, q=0.005:0.00353505, q=0.01:0.00707007, q=0.05:0.0388329, q=0.1:0.0734346
- peierls + q0_mass_diagnostic + minus: q=0.001:0.000771518, q=0.005:0.00385813, q=0.01:0.00772246, q=0.05:0.0639304, q=0.1:0.100674
- midpoint + q0_mass_diagnostic + minus: q=0.001:0.000771518, q=0.005:0.00385813, q=0.01:0.00772246, q=0.05:0.0639331, q=0.1:0.100668
- midpoint + q0_mass_diagnostic + plus: q=0.001:0.000707012, q=0.005:0.00353505, q=0.01:0.00707008, q=0.05:0.0388358, q=0.1:0.0734551

## Comparison
- Peierls current vertex does not materially lower the max Ward residual in this prototype (midpoint/Peierls factor = 1); possible reasons include the missing contact term or remaining vertex/contact closure gaps.
- q0_mass_diagnostic lowers the full-grid Peierls Ward residual most for plus sign by a factor of 1.09123.
- For q_model <= 0.01, q0_mass_diagnostic does not materially improve the Peierls residual; a complete finite-q Peierls contact may still be required.
- q0_mass_diagnostic, even when helpful at small q, is not a complete finite-q Peierls contact.
- A final gauge-consistent finite-q response still requires a contact term from the same finite-q Peierls expansion.

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/data/normal_ward_identity_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/left_right_ward_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q_by_vertex_scheme.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q_by_contact_scheme.png
