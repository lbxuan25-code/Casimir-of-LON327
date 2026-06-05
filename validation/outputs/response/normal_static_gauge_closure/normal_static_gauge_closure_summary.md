# normal-state static gauge closure 诊断

这是 normal-state static gauge closure 诊断。它使用 Peierls-twist finite-difference
free-energy stiffness 作为独立 baseline，用于检查 normal kernel convention。
Peierls baseline 是 stiffness reference；本诊断不假设 clean normal-state
stiffness 在有限 mesh 上必须为零。

本诊断的目的不是选择最终 response 公式，而是定位 static closure failure 是否
来自 normal K_para 符号、K_dia 符号/contact convention、mass operator 或
intra/inter balance。

本诊断不修改正式 response 公式，不修改 BdG、Casimir、reflection 或 finite-q
代码，也不是最终 optical conductivity 或 Casimir 输入。

run_command = `python validation/scripts/numerical_stability/diagnose_normal_static_gauge_closure.py --omega-list 0.0 1e-06 1e-05 0.0001 --nk-list 8 12 16 24 --temperature 30.0 --eta 0.0001 --twist-list 0.001 0.0005 0.0002 --output-prefix validation/outputs/response/normal_static_gauge_closure/data/normal_static_gauge_closure`
quick_test_only=False
benchmark_only=True
local_response=True
normal_static_gauge_closure_diagnostic=True
peierls_twist_diagnostic=True
not_final_response_formula=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

## 参数
- omega_list=0, 1e-06, 1e-05, 0.0001
- nk_list=8, 12, 16, 24
- temperature_K=30
- eta_eV=0.0001
- twist_list=0.001, 0.0005, 0.0002

## omega=0、twist=0.0002 时的 Peierls D_fd 趋势
- nk=8: D_fd_xx=0.416418
- nk=12: D_fd_xx=0.336948
- nk=16: D_fd_xx=0.348808
- nk=24: D_fd_xx=0.355277

## 候选 Convention
dominant_best_candidate=minus_para_plus_dia
minus_para_plus_dia 表示 K_dia - K_para。
largest_mean_static_component_norm=K_dia

## 下一步
使用 Peierls baseline 和 intra/inter/dia 分解判断哪个 normal-state convention
需要解析复查。没有推导前，不要把这里报告的 best candidate 当作公式修正。

## 图像
- validation/outputs/response/normal_static_gauge_closure/figures/D_fd_xx_vs_nk.png
- validation/outputs/response/normal_static_gauge_closure/figures/candidate_K_xx_vs_nk.png
- validation/outputs/response/normal_static_gauge_closure/figures/candidate_error_vs_nk.png
- validation/outputs/response/normal_static_gauge_closure/figures/intra_inter_dia_decomposition_vs_nk.png
- validation/outputs/response/normal_static_gauge_closure/figures/best_candidate_error_vs_omega.png
