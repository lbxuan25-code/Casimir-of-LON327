# Spatial-current Ward sector audit

This is a spatial-current Ward-sector audit.
It is not conductivity, not a reflection/Casimir input, and not a material conclusion.
No Peierls current/contact vertex formula is modified here.

run_command = `python validation/scripts/response/audit_spatial_current_ward_sector.py`
quick_mode=False
expanded_data_written=False
response_computed=True
conductivity_computed=False
casimir_computed=False
normal_state_only=True
bdg_computed=False
not_final_finite_q_conductivity=True
not_final_casimir_conclusion=True

## Spatial residual maximum
- max combo = peierls+q0_mass_diagnostic+plus
- side = left
- component = y
- orientation = longitudinal
- matsubara_n = 1
- nk = 12
- q_model = 0.1
- q_angle = 1.5708
- max residual_abs = 0.122146
- left/right residual max ratio = 1
- left/right spatial residuals are close.

## Small-q scaling
- alpha is fitted from max residual_abs at q_model = 0.001, 0.005, 0.01.
- peierls+none: alpha=0.999998 (approximately O(q), spatial sector is the full Ward O(q) source)
- peierls+q0_mass_diagnostic+plus: alpha=1 (approximately O(q), spatial sector is the full Ward O(q) source)
- peierls+q0_mass_diagnostic+minus: alpha=0.999996 (approximately O(q), spatial sector is the full Ward O(q) source)
- peierls+finite_q_peierls+plus: alpha=0.999996 (approximately O(q), spatial sector is the full Ward O(q) source)
- peierls+finite_q_peierls+minus: alpha=0.999988 (approximately O(q), spatial sector is the full Ward O(q) source)
- midpoint+none: alpha=1 (approximately O(q), spatial sector is the full Ward O(q) source)

## Contact comparison
- peierls+none max residual_abs = 0.0699203
- q0_mass_diagnostic best sign = minus; max residual_abs = 0.0417792
- finite_q_peierls best sign = minus; max residual_abs = 0.0416971
- peierls+none / q0_mass_diagnostic best residual factor = 1.67357
- peierls+none / finite_q_peierls best residual factor = 1.67686
- q0_best / finite_best residual factor = 1.00197
- finite_q_peierls is not materially better than q0_mass_diagnostic at the 1% threshold.
- strongest finite_q_peierls+plus contact_effect_ratio occurs in transverse; value = 0.356425
- Contact improves middle/large q but does not remove the small-q O(q) term; the form factor is not the O(q) closure issue.

## Next checks
- finite_q_peierls minus is better; prioritize checking contact sign convention.
- If longitudinal alone is bad, inspect longitudinal current/contact and density-current relations.
- If transverse is also bad, inspect the current-current bubble convention as a whole.

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/spatial_current_ward_sector/data/spatial_current_ward_sector_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/spatial_current_ward_sector/figures/spatial_residual_vs_q.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/spatial_current_ward_sector/figures/spatial_closure_ratio_by_component.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/spatial_current_ward_sector/figures/contact_effect_ratio_vs_q.png
