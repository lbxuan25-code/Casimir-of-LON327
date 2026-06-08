# Casimir q-grid to model-q unit/sampling audit

This is a unit/sampling audit only.
No response tensor is computed.
No finite-q conductivity is produced.
No Casimir conclusion is made.

run_command = `python validation/scripts/units/audit_casimir_q_grid_to_model_q.py`
quick_mode = False
in_plane_lattice_constant_list_m = 3.75e-10 3.85e-10 3.9e-10 3.95e-10
a_parallel is the in-plane pseudotetragonal / Ni-Ni effective lattice constant used to convert q_SI to model-q units.
This is a configurable in-plane conversion length, not a final crystallographic refinement parameter.
distance_list_m = 3e-08 5e-08 7.5e-08 1e-07 1.5e-07 2e-07
u_max = 80
du = 0.5
phi_num = 32
small_q_threshold_list = 0.001 0.005 0.01 0.05 0.1
expanded_data_written=False

## Scope flags
unit_audit_only=True
response_computed=False
casimir_computed=False
not_final_casimir_conclusion=True

## Full grid q_model range
- q_model_min = 0
- q_model_max = 1.05333
- q_model_max/pi = 0.335286
- q_model_max/(2pi) = 0.167643

## a_parallel sensitivity
- a_parallel = 3.75e-10 m (3.75 A): q_model_max = 1, q_model_max/pi = 0.31831, q_model_max/(2pi) = 0.159155
- a_parallel = 3.85e-10 m (3.85 A): q_model_max = 1.02667, q_model_max/pi = 0.326798, q_model_max/(2pi) = 0.163399
- a_parallel = 3.9e-10 m (3.9 A): q_model_max = 1.04, q_model_max/pi = 0.331042, q_model_max/(2pi) = 0.165521
- a_parallel = 3.95e-10 m (3.95 A): q_model_max = 1.05333, q_model_max/pi = 0.335286, q_model_max/(2pi) = 0.167643
- q_model,max remains O(1), so Stage 1 q<=0.005 still only tests small-q limit.

## q_model_max by distance
- d = 3e-08 m: q_model_max = 1.05333, q_model_max/pi = 0.335286, q_model_max/(2pi) = 0.167643
- d = 5e-08 m: q_model_max = 0.632, q_model_max/pi = 0.201172, q_model_max/(2pi) = 0.100586
- d = 7.5e-08 m: q_model_max = 0.421333, q_model_max/pi = 0.134115, q_model_max/(2pi) = 0.0670573
- d = 1e-07 m: q_model_max = 0.316, q_model_max/pi = 0.100586, q_model_max/(2pi) = 0.050293
- d = 1.5e-07 m: q_model_max = 0.210667, q_model_max/pi = 0.0670573, q_model_max/(2pi) = 0.0335286
- d = 2e-07 m: q_model_max = 0.158, q_model_max/pi = 0.050293, q_model_max/(2pi) = 0.0251465

## Small-q coverage
- threshold q <= 0.001: 0.724638% (896/123648 sampled points)
- threshold q <= 0.005: 1.915114% (2368/123648 sampled points)
- threshold q <= 0.01: 3.571429% (4416/123648 sampled points)
- threshold q <= 0.05: 16.485507% (20384/123648 sampled points)
- threshold q <= 0.1: 32.712215% (40448/123648 sampled points)

## Stage 1 coverage check
- Stage 1 sampled q_model list found in repository: 0 0.0001 0.0002 0.0005 0.001 0.002 0.005
- Stage 1 q_model_max = 0.005
- Current audit q_model_max = 1.05333
- Stage 1 sampled q range does not cover the current Casimir-relevant q_model range; it only tests the small-q limit.

## Stage 3 recommended q-list
- small-q regression list: 0 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01
- Casimir-relevant q list: 0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 0.75 1 1.1
- BZ stress list: 0.392699 0.785398 1.5708 3.14159 6.28319

The BZ stress list is for numerical stress testing only; it is not a statement that the audited local Casimir grid reaches those momenta.

## Output files
- compact CSV: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/data/casimir_q_grid_model_q_audit_compact.csv
- expanded_data_written=False
- expanded CSV/NPZ not written; rerun with --write-expanded-data to generate them locally.
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/figures/q_model_max_vs_distance.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/figures/q_model_histogram.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/figures/q_model_coverage_by_distance.png
- figure: /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/units/casimir_q_grid_model_q_audit/figures/q_model_max_vs_a_parallel.png
