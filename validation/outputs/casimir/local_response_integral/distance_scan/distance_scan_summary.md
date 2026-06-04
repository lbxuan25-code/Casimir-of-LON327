# Local-response distance scan benchmark 摘要

本脚本用于在已通过数值稳定性检查的推荐参数下，扫描距离 d，建立当前
local-response zero-torque baseline 的距离依赖基准。它只复用已有
local-response integral benchmark，不引入真实各向异性机制，也不输出正式
Casimir torque 结论。

full_run_command = `python validation/scripts/casimir/benchmark_casimir_local_response_distance_scan.py --kinds normal spm dwave --distance-list 3e-08 5e-08 7.5e-08 1e-07 1.5e-07 2e-07 --theta-list 0 0.3926990817 0.7853981634 1.1780972451 1.5707963268 --matsubara-max 64 --u-max 80 --du 0.5 --phi-num 32 --temperature 30 --normal-nk 96 --normal-eta 0.0001 --normal-sampling fs_adaptive --normal-refine-factor 8 --bdg-nk 32 --delta0 0.04 --cache-dir validation/cache/casimir_local_response/response_tensors --output-prefix validation/outputs/casimir/local_response_integral/distance_scan/data/distance_scan --use-response-cache --include-toy-anisotropic-control --rebuild-response-cache --progress`
quick_test_only=False
full_distance_scan_completed=True
response_cache_used=True
response_cache_entries=256
response_cache_rebuilt=True
response_cache_hits=1280
response_cache_misses=256
response_cache_writes=256
local_response=True
finite_momentum_resolved=False
n0_policy=skip
benchmark_only=True
not_final_Casimir_conclusion=True

## 推荐参数
- matsubara_max=64
- u_max=80
- du=0.5
- kparallel_num=161
- phi_num=32
- normal_sampling=fs_adaptive
- normal_nk=96
- normal_refine_factor=8
- bdg_nk=32
- delta0=0.04

## 距离扫描范围
- 3e-08, 5e-08, 7.5e-08, 1e-07, 1.5e-07, 2e-07

## zero-torque baseline
- torque_tolerance=1e-20
- normal: zero_torque_baseline=True, max_abs_torque=1.34809e-22
- spm: zero_torque_baseline=True, max_abs_torque=3.37024e-23
- dwave: zero_torque_baseline=True, max_abs_torque=5.05536e-23

## toy anisotropic control
toy_anisotropic_control_enabled=True
toy_anisotropic_control_passed=True

## 结论边界
当前仍不是正式 Casimir 结论，原因是：
- local_response=True
- finite_momentum_resolved=False
- n0_policy=skip
- benchmark_only=True
local_response_distance_scan_benchmark_ready=True
ready_for_anisotropy_mechanism_benchmark=True
not_final_Casimir_conclusion=True
