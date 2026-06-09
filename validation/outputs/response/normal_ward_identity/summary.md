# Normal-state Pi_mu_nu Ward identity prototype

This is a normal-state Pi_mu_nu Ward diagnostic.
It compares midpoint velocity, Peierls current vertex, q=0 mass contact, and finite-q Peierls contact schemes.
It is not conductivity and not a reflection/Casimir input.
The finite_q_peierls contact is connected to this Ward diagnostic, but this is still not final conductivity.
Response-level sign, normalization, equal-time term, and density-vertex conventions still need final closure.
Large Ward residuals may reflect finite-q vertex/contact-term closure gaps, not a material conclusion.

run_command = `python validation/scripts/response/diagnose_normal_ward_identity.py`
quick_mode=False
expanded_data_written=False
density_current_included=True
current_current_included=True
diamagnetic_contact_included=True only for contact_scheme=q0_mass_diagnostic or finite_q_peierls
contact_scheme=none or q0_mass_diagnostic or finite_q_peierls
not_final_finite_q_contact=True
normal_state_only=True
bdg_computed=False
conductivity_computed=False
casimir_computed=False
not_final_casimir_conclusion=True
not_final_finite_q_conductivity=True

## Parameter grid
- vertex_schemes = midpoint peierls
- contact_schemes = none q0_mass_diagnostic finite_q_peierls
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
- peierls + finite_q_peierls + plus: max left Ward error = 0.0734155
- peierls + finite_q_peierls + plus: max right Ward error = 0.0734155
- peierls + finite_q_peierls + plus: max Ward error = 0.0734155
- peierls + finite_q_peierls + plus: max Ward error for q_model <= 0.01 = 0.00707007
- peierls + finite_q_peierls + minus: max left Ward error = 0.100663
- peierls + finite_q_peierls + minus: max right Ward error = 0.100663
- peierls + finite_q_peierls + minus: max Ward error = 0.100663
- peierls + finite_q_peierls + minus: max Ward error for q_model <= 0.01 = 0.00772247
- midpoint + finite_q_peierls + minus: max left Ward error = 0.100658
- midpoint + finite_q_peierls + minus: max right Ward error = 0.100658
- midpoint + finite_q_peierls + minus: max Ward error = 0.100658
- midpoint + finite_q_peierls + minus: max Ward error for q_model <= 0.01 = 0.00772247
- midpoint + finite_q_peierls + plus: max left Ward error = 0.073436
- midpoint + finite_q_peierls + plus: max right Ward error = 0.073436
- midpoint + finite_q_peierls + plus: max Ward error = 0.073436
- midpoint + finite_q_peierls + plus: max Ward error for q_model <= 0.01 = 0.00707007
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
- peierls + finite_q_peierls + plus: q=0.001:0.000707012, q=0.005:0.00353505, q=0.01:0.00707007, q=0.05:0.0388305, q=0.1:0.0734155
- peierls + finite_q_peierls + minus: q=0.001:0.000771518, q=0.005:0.00385813, q=0.01:0.00772247, q=0.05:0.0639362, q=0.1:0.100663
- midpoint + finite_q_peierls + minus: q=0.001:0.000771518, q=0.005:0.00385813, q=0.01:0.00772247, q=0.05:0.0639388, q=0.1:0.100658
- midpoint + finite_q_peierls + plus: q=0.001:0.000707012, q=0.005:0.00353505, q=0.01:0.00707007, q=0.05:0.0388333, q=0.1:0.073436
- midpoint + q0_mass_diagnostic + minus: q=0.001:0.000771518, q=0.005:0.00385813, q=0.01:0.00772246, q=0.05:0.0639331, q=0.1:0.100668
- midpoint + q0_mass_diagnostic + plus: q=0.001:0.000707012, q=0.005:0.00353505, q=0.01:0.00707008, q=0.05:0.0388358, q=0.1:0.0734551

## Comparison
- Peierls current vertex does not materially lower the max Ward residual in this prototype (midpoint/Peierls factor = 1); possible reasons include the missing contact term or remaining vertex/contact closure gaps.
- q0_mass_diagnostic lowers the full-grid Peierls Ward residual most for plus sign by a factor of 1.09123.
- For q_model <= 0.01, q0_mass_diagnostic does not materially improve the Peierls residual; a complete finite-q Peierls contact may still be required.
- finite_q_peierls contact is best with plus sign on the full grid; Peierls-none / finite_q_peierls factor = 1.09152.
- finite_q_peierls and q0_mass_diagnostic give similar full-grid Ward residuals within a 1% comparison threshold.
- For q_model <= 0.01, finite_q_peierls does not materially improve the Peierls residual.
- For q_model >= 0.05, finite_q_peierls improves most for plus sign by a factor of 1.09152.
- If finite_q_peierls does not close the Ward residual, likely causes include contact sign/normalization still being inconsistent at response level, missing equal-time or density-vertex pieces, or the need for a stricter response-level Ward derivation.

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/data/normal_ward_identity_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/left_right_ward_error_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q_by_vertex_scheme.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_identity/figures/ward_error_vs_q_by_contact_scheme.png
