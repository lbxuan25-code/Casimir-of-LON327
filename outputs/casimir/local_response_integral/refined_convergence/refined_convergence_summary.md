# Refined Local-Response Convergence Summary

This is a refined convergence benchmark, not a final Casimir conclusion.
old_cutoff_scan_issue = fixed kparallel_num caused changing du when cutoff increased
new_cutoff_scan = u=k_parallel*d with fixed du
full_run_command = `python scripts/refine_casimir_local_convergence_blockers.py --kinds normal spm dwave --distance 5e-08 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --energy-theta-list 0 --torque-check-theta-list 0 0.7853981634 1.5707963268 --u-max-list 20 40 60 80 --du 0.5 --matsubara-max-list 24 32 48 64 --temperature 30 --normal-nk 96 --normal-eta 0.0001 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --phi-num 32 --output-prefix outputs/casimir/local_response_integral/refined_convergence/data/refined_local_convergence --cache-dir outputs/casimir/local_response_integral/cache --use-response-cache`
quick_test_only = False
full_run_completed = True
response_cache_used=True
response_cache_entries=192
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
- cutoff/normal: u_max=80, du=0.5, implied_kparallel_num=161, tail_shell_indicator=0, last_two_relative_change=0, cutoff_status=candidate_converged
- cutoff/spm: u_max=80, du=0.5, implied_kparallel_num=161, tail_shell_indicator=0, last_two_relative_change=0, cutoff_status=candidate_converged
- cutoff/dwave: u_max=80, du=0.5, implied_kparallel_num=161, tail_shell_indicator=0, last_two_relative_change=0, cutoff_status=candidate_converged

## Matsubara Scan
- matsubara/normal: matsubara_max=64, matsubara_tail_indicator=0.000576396, last_two_relative_change=0.0146604, matsubara_status=candidate_converged
- matsubara/spm: matsubara_max=64, matsubara_tail_indicator=0.000880702, last_two_relative_change=0.0229743, matsubara_status=loose_converged
- matsubara/dwave: matsubara_max=64, matsubara_tail_indicator=0.000877982, last_two_relative_change=0.022923, matsubara_status=loose_converged

## Baseline
clean_cutoff_converged = True
extended_matsubara_converged = True
zero_torque_baseline = True
warning_possible_spurious_torque = False
normal_max_abs_torque_over_theta = 3.37024e-23
spm_max_abs_torque_over_theta = 1.68512e-23
dwave_max_abs_torque_over_theta = 3.37024e-23
can_return_to_local_response_distance_scan_benchmark = True
local_integral_benchmark_ready_for_distance_scan = True
not final Casimir conclusion
