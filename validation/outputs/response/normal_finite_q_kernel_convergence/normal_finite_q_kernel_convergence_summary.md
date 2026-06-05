# Normal finite-q kernel convergence diagnostic

本脚本只测试 normal finite-q current-current kernel (K)。
q=0 与 q!=0 都通过同一 K 接口 normal_current_current_kernel_imag_axis 计算。
q=0 分支不再调用 public local sigma。
默认只测 n>=1 positive Matsubara；本阶段不处理 n=0 true static。
Matsubara 频率直接使用 bosonic_matsubara_energy_eV(n, temperature_K)；不使用 omega+eta 频率展宽。
current-current-only 不是 gauge-closed finite-q conductivity。
Ward identity 尚未检查。
本脚本不修改 BdG、Casimir、reflection matrix。
本脚本不输出最终 finite-q conductivity 或 Casimir 结论。

run_command = `python validation/scripts/numerical_stability/diagnose_normal_finite_q_response.py --matsubara-n-list 1 2 4 8 --temperature 30 --q-list 0 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 --q-angle-list 0 pi/8 pi/4 '3*pi/8' pi/2 --nk-list 16 24 32 --output-prefix validation/outputs/response/normal_finite_q_kernel_convergence/data/normal_finite_q_kernel_convergence_full`
quick_mode=False
finite_momentum_resolved=True
normal_state=True
current_current_kernel_only=True
midpoint_vertex_approximation=True
not_peierls_exact_vertex=True
ward_identity_not_yet_checked=True
not_final_casimir_input=True

## Quick diagnostic status
- q=0 same-interface error: 0
- smallest sampled nonzero q: 0.0001
- maximum same-interface error at smallest nonzero q: 0.000205641
- maximum C4 covariance error: 6.58137e-15
- all K components finite: True
