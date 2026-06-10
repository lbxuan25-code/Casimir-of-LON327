# Normal Ward residual decomposition

This is a Ward decomposition diagnostic.
It is not conductivity, not a reflection/Casimir input, and not a material conclusion.
The rows decompose each left/right component into iOmega, qx, and qy terms using a local component scale.

run_command = `python validation/scripts/response/decompose_normal_ward_residual.py`
quick_mode=False
expanded_data_written=False
response_computed=True
conductivity_computed=False
casimir_computed=False
normal_state_only=True
bdg_computed=False
not_final_finite_q_conductivity=True
not_final_casimir_conclusion=True

## Maxima
- max residual_abs: combo=peierls+q0_mass_diagnostic+plus; side=left; component=y; q=0.1; value=0.122146
- max closure_ratio: combo=midpoint+none; side=left; component=0; q=0.001; value=1
- density component max closure_ratio = 1
- spatial component max closure_ratio = 1
- density component max residual_abs = 0.074125
- spatial component max residual_abs = 0.122146
- left max closure_ratio = 1
- right max closure_ratio = 1
- left/right max ratio = 1
- left/right structures are close.

## Contact/component comparison
- density component is already comparable to or larger than spatial residuals.
- component 0: peierls+none / finite_q_peierls+plus closure factor = 1
- component x: peierls+none / finite_q_peierls+plus closure factor = 0.99909
- component y: peierls+none / finite_q_peierls+plus closure factor = 1.00001
- spatial closure q0_mass_diagnostic+plus / finite_q_peierls+plus factor = 1.00001
- finite_q_peierls minus/plus max closure ratio factor = 1

## Small-q scaling
- alpha is fitted from max residual_abs at q_model = 0.001, 0.005, 0.01.
- midpoint+none: alpha = 1 (approximately O(q), suggesting a response-level O(q) gap)
- peierls+none: alpha = 0.999998 (approximately O(q), suggesting a response-level O(q) gap)
- peierls+q0_mass_diagnostic+plus: alpha = 1 (approximately O(q), suggesting a response-level O(q) gap)
- peierls+finite_q_peierls+plus: alpha = 0.999996 (approximately O(q), suggesting a response-level O(q) gap)
- peierls+finite_q_peierls+minus: alpha = 0.999988 (approximately O(q), suggesting a response-level O(q) gap)

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_residual_decomposition/data/normal_ward_residual_decomposition_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_residual_decomposition/figures/max_closure_ratio_by_component.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/normal_ward_residual_decomposition/figures/q_scaling_by_combo.png
