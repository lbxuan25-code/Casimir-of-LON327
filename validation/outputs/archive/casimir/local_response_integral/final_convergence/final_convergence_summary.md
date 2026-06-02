# Final Local-Response Convergence Summary

full_run_command = `python validation/scripts/casimir/run_casimir_local_convergence_final.py --kinds normal spm dwave --distance 5e-08 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --matsubara-max-list 4 8 16 24 --kparallel-num-list 32 64 96 --kparallel-max-factor-list 20 40 60 --phi-num-list 32 64 96 --temperature 30 --normal-nk 96 --normal-eta 0.0001 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --output-prefix validation/outputs/archive/casimir/local_response_integral/final_convergence/data/final_local_convergence`
quick_test_result = False
full_convergence_result = available
full_run_pending = False
full_run_completed = True
local_response=True
finite_q_resolved=False
n0_policy=skip
benchmark_only=True
not_final_casimir_conclusion=True

## Scan Status
- matsubara/normal: last_two_relative_change=0.124614, convergence_status=not_converged
- matsubara/spm: last_two_relative_change=0.179666, convergence_status=not_converged
- matsubara/dwave: last_two_relative_change=0.179678, convergence_status=not_converged
- kparallel_num/normal: last_two_relative_change=0.0120053, convergence_status=candidate_converged
- kparallel_num/spm: last_two_relative_change=0.017628, convergence_status=candidate_converged
- kparallel_num/dwave: last_two_relative_change=0.0176322, convergence_status=candidate_converged
- kparallel_cutoff/normal: last_two_relative_change=0.788334, convergence_status=not_converged
- kparallel_cutoff/spm: last_two_relative_change=0.824043, convergence_status=not_converged
- kparallel_cutoff/dwave: last_two_relative_change=0.82406, convergence_status=not_converged
- phi/normal: last_two_relative_change=1.60128e-16, convergence_status=candidate_converged
- phi/spm: last_two_relative_change=1.4382e-16, convergence_status=candidate_converged
- phi/dwave: last_two_relative_change=0, convergence_status=candidate_converged

## Diagnostics
matsubara_tail_indicator_max = 0.241453
normal_max_abs_torque_over_theta = 3.37024e-23
spm_max_abs_torque_over_theta = 3.37024e-23
dwave_max_abs_torque_over_theta = 8.42559e-24
zero_torque_baseline = True
warning_possible_spurious_torque = False
can_enter_next_stage = False
local_integral_benchmark_ready_for_distance_scan = False
not final Casimir conclusion
