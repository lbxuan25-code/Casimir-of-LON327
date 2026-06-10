# Density-current Ward sector audit

This is a density-current Ward-sector audit.
It is not conductivity, not a reflection/Casimir input, and not a material conclusion.
Only the density residuals R_L[0] and R_R[0] are analyzed.

run_command = `python validation/scripts/response/audit_density_current_ward_sector.py`
quick_mode=False
expanded_data_written=False
response_computed=True
conductivity_computed=False
casimir_computed=False
normal_state_only=True
bdg_computed=False
not_final_finite_q_conductivity=True
not_final_casimir_conclusion=True

## Density residual maxima
- max density residual side = left
- max combo = midpoint+none
- matsubara_n = 4
- nk = 12
- q_model = 0.1
- q_angle = 0.785398
- max residual_abs = 0.074125
- left/right residual max ratio = 1
- left/right density residuals are close.

## Contact sensitivity
- contact_sensitive_warning = False
- density residual is not contact-sensitive at the 1% threshold, as expected because contact enters only the spatial-spatial block.
- peierls+none / peierls+finite_q_peierls+plus max density residual factor = 1
- finite_q_peierls minus/plus density residual factor = 1

## Small-q scaling
- alpha is fitted from q_model = 0.001, 0.005, 0.01.
- peierls+none: left alpha=1.99883, right alpha=1.99883 (approximately O(q^2), so this density sector is unlikely to be the dominant O(q) gap)
- peierls+q0_mass_diagnostic+plus: left alpha=1.99883, right alpha=1.99883 (approximately O(q^2), so this density sector is unlikely to be the dominant O(q) gap)
- peierls+finite_q_peierls+plus: left alpha=1.99883, right alpha=1.99883 (approximately O(q^2), so this density sector is unlikely to be the dominant O(q) gap)
- peierls+finite_q_peierls+minus: left alpha=1.99883, right alpha=1.99883 (approximately O(q^2), so this density sector is unlikely to be the dominant O(q) gap)
- midpoint+none: left alpha=1.99883, right alpha=1.99883 (approximately O(q^2), so this density sector is unlikely to be the dominant O(q) gap)

## Likely next checks
- Density-sector residual is contact-insensitive and closer to O(q^2), so the O(q) gap seen in the full decomposition is more likely tied to spatial current/equal-time or Kubo convention closure; still keep Gamma0 embedding as a later consistency check.

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/density_current_ward_sector/data/density_current_ward_sector_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/density_current_ward_sector/figures/density_residual_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/density_current_ward_sector/figures/density_closure_ratio_vs_q.png
