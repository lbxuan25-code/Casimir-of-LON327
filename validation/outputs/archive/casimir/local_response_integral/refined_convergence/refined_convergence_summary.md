# Refined Local-Response Convergence Summary

This is a refined convergence benchmark, not a final Casimir conclusion.
old_cutoff_scan_issue = fixed kparallel_num caused changing du when cutoff increased
new_cutoff_scan = u=k_parallel*d with fixed du
full_run_command = `python validation/scripts/casimir/refine_casimir_local_convergence_blockers.py --kinds normal spm dwave --distance 5e-08 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --energy-theta-list 0 --torque-check-theta-list 0 0.7853981634 1.5707963268 --u-max-list 20 40 60 80 --du 0.5 --matsubara-max-list 24 32 48 64 --temperature 30 --normal-nk 96 --normal-eta 0.0001 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --phi-num 32 --output-prefix validation/outputs/archive/casimir/local_response_integral/refined_convergence/data/refined_local_convergence --cache-dir validation/cache/casimir_local_response/response_tensors --use-response-cache`
quick_test_only = True
full_run_completed = False
response_cache_used=False
response_cache_entries=0
response_cache_rebuilt=False
response_cache_hits=0
response_cache_misses=0
response_cache_writes=0
local_response=True
finite_q_resolved=False
n0_policy=skip
benchmark_only=True
not_final_casimir_conclusion=True

## Cutoff Scan
- cutoff/normal: u_max=6, du=2, implied_kparallel_num=4, tail_shell_indicator=0.0189868, last_two_relative_change=0.0189868, cutoff_status=candidate_converged
- cutoff/spm: u_max=6, du=2, implied_kparallel_num=4, tail_shell_indicator=0.0180216, last_two_relative_change=0.0180216, cutoff_status=candidate_converged
- cutoff/dwave: u_max=6, du=2, implied_kparallel_num=4, tail_shell_indicator=0.0180214, last_two_relative_change=0.0180214, cutoff_status=candidate_converged

## Matsubara Scan
- matsubara/normal: matsubara_max=2, matsubara_tail_indicator=0.468653, last_two_relative_change=0.468653, matsubara_status=matsubara_not_converged
- matsubara/spm: matsubara_max=2, matsubara_tail_indicator=0.498836, last_two_relative_change=0.498836, matsubara_status=matsubara_not_converged
- matsubara/dwave: matsubara_max=2, matsubara_tail_indicator=0.498842, last_two_relative_change=0.498842, matsubara_status=matsubara_not_converged

## Baseline
clean_cutoff_converged = True
extended_matsubara_converged = False
zero_torque_baseline = True
warning_possible_spurious_torque = False
normal_max_abs_torque_over_theta = 0
spm_max_abs_torque_over_theta = 0
dwave_max_abs_torque_over_theta = 2.1064e-24
can_return_to_local_response_distance_scan_benchmark = False
local_integral_benchmark_ready_for_distance_scan = False
not final Casimir conclusion
quick_test_only=True
no_full_convergence_conclusion=True
full_run_pending_user_terminal=True
