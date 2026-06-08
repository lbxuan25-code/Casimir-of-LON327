# Casimir q-grid to model-q unit/sampling audit

This is a unit/sampling audit only.
No response tensor is computed.
No finite-q conductivity is produced.
No Casimir conclusion is made.

run_command = `python validation/scripts/units/audit_casimir_q_grid_to_model_q.py`
quick_mode = False
lattice_constant_m = 3.9e-10
lattice_constant_m is a configurable assumption for this audit and is not a final material parameter.
distance_list_m = 3e-08 5e-08 7.5e-08 1e-07 1.5e-07 2e-07
u_max = 80
du = 0.5
phi_num = 32
small_q_threshold_list = 0.001 0.005 0.01 0.05 0.1

## Scope flags
unit_audit_only=True
response_computed=False
casimir_computed=False
not_final_casimir_conclusion=True

## Full grid q_model range
- q_model_min = 0
- q_model_max = 1.04
- q_model_max/pi = 0.331042
- q_model_max/(2pi) = 0.165521

## q_model_max by distance
- d = 3e-08 m: q_model_max = 1.04, q_model_max/pi = 0.331042, q_model_max/(2pi) = 0.165521
- d = 5e-08 m: q_model_max = 0.624, q_model_max/pi = 0.198625, q_model_max/(2pi) = 0.0993127
- d = 7.5e-08 m: q_model_max = 0.416, q_model_max/pi = 0.132417, q_model_max/(2pi) = 0.0662085
- d = 1e-07 m: q_model_max = 0.312, q_model_max/pi = 0.0993127, q_model_max/(2pi) = 0.0496563
- d = 1.5e-07 m: q_model_max = 0.208, q_model_max/pi = 0.0662085, q_model_max/(2pi) = 0.0331042
- d = 2e-07 m: q_model_max = 0.156, q_model_max/pi = 0.0496563, q_model_max/(2pi) = 0.0248282

## Small-q coverage
- threshold q <= 0.001: 0.724638% (224/30912 sampled points)
- threshold q <= 0.005: 1.863354% (576/30912 sampled points)
- threshold q <= 0.01: 3.519669% (1088/30912 sampled points)
- threshold q <= 0.05: 16.356108% (5056/30912 sampled points)
- threshold q <= 0.1: 32.401656% (10016/30912 sampled points)

## Stage 1 coverage check
- Stage 1 sampled q_model list found in repository: 0 0.0001 0.0002 0.0005 0.001 0.002 0.005
- Stage 1 q_model_max = 0.005
- Current audit q_model_max = 1.04
- Stage 1 sampled q range does not cover the current Casimir-relevant q_model range; it only tests the small-q limit.

## Stage 3 recommended q-list
- small-q regression list: 0 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01
- Casimir-relevant q list: 0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.75 1 1.04
- BZ stress list: 0.392699 0.785398 1.5708 3.14159 6.28319

The BZ stress list is for numerical stress testing only; it is not a statement that the audited local Casimir grid reaches those momenta.

## Output files
- CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/data/casimir_q_grid_model_q_audit.csv
- NPZ: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/data/casimir_q_grid_model_q_audit.npz
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/figures/q_model_max_vs_distance.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/figures/q_model_histogram.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/figures/q_model_coverage_by_distance.png
