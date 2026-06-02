# Final Local-Response Convergence Summary

full_run_command = `python validation/scripts/casimir/run_casimir_local_convergence_final.py --kinds normal spm dwave --distance 5e-08 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --matsubara-max-list 4 8 16 24 --kparallel-num-list 32 64 96 --kparallel-max-factor-list 20 40 60 --phi-num-list 32 64 96 --temperature 30 --normal-nk 96 --normal-eta 0.0001 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --output-prefix validation/outputs/archive/casimir/local_response_integral/final_convergence/data/final_local_convergence`
quick_test_result = True
full_convergence_result = not_available
full_run_pending = True
full_run_completed = False
local_response=True
finite_q_resolved=False
n0_policy=skip
benchmark_only=True
not_final_casimir_conclusion=True

## Scan Status
- matsubara/normal: last_two_relative_change=0.47748, convergence_status=not_converged
- matsubara/spm: last_two_relative_change=0.499179, convergence_status=not_converged
- matsubara/dwave: last_two_relative_change=0.499183, convergence_status=not_converged
- kparallel_num/normal: last_two_relative_change=0.797718, convergence_status=not_converged
- kparallel_num/spm: last_two_relative_change=0.801984, convergence_status=not_converged
- kparallel_num/dwave: last_two_relative_change=0.801985, convergence_status=not_converged
- kparallel_cutoff/normal: last_two_relative_change=3.65486, convergence_status=not_converged
- kparallel_cutoff/spm: last_two_relative_change=3.9764, convergence_status=not_converged
- kparallel_cutoff/dwave: last_two_relative_change=3.97647, convergence_status=not_converged
- phi/normal: last_two_relative_change=0, convergence_status=candidate_converged
- phi/spm: last_two_relative_change=0, convergence_status=candidate_converged
- phi/dwave: last_two_relative_change=1.44823e-16, convergence_status=candidate_converged

## Diagnostics
matsubara_tail_indicator_max = 0.499415
normal_max_abs_torque_over_theta = 2.1064e-24
spm_max_abs_torque_over_theta = 0
dwave_max_abs_torque_over_theta = 0
zero_torque_baseline = True
warning_possible_spurious_torque = False
can_enter_next_stage = False
local_integral_benchmark_ready_for_distance_scan = False
not final Casimir conclusion
full_run_pending_user_terminal=True
no_full_convergence_conclusion=True
