# BdG 正常极限 kernel 分解诊断

这是 BdG normal-limit kernel 分解诊断。它在同一 mesh 和 KuboConfig 下，
比较 Delta0=0 的 BdG K_para、K_dia、K_total，以及本地构造的 normal-state
kernel-level K_para 和 mass-expectation K_dia。

在当前 positive-bubble convention 下，K_total 解释为已经由 Peierls/free-energy
验证过的 stiffness kernel K_dia - K_para。

本诊断用于定位 static stiffness mismatch 是否来自 paramagnetic bubble、
diamagnetic/contact term、符号 convention、Nambu redundancy 或 occupation
convention。它不是最终 response 公式选择。

本诊断不修改正式 BdG response 公式，不修改 Casimir 计算，不包含 finite
momentum response，也不是最终 optical conductivity 或 Casimir 输入。

run_command = `python validation/scripts/numerical_stability/diagnose_bdg_normal_limit_kernel_decomposition.py --kinds spm dwave --omega-list 0.0 1e-06 2e-06 5e-06 1e-05 2e-05 5e-05 0.0001 --nk 16 --temperature 30.0 --eta 0.0001 --output-prefix /home/liubx25/Ni_Research/Projects/Casimir_Torque_of_LNO327/validation/outputs/response/bdg_normal_limit_kernel_decomposition/data/bdg_normal_limit_kernel_decomposition`
quick_test_only=False
benchmark_only=True
local_response=True
normal_limit_decomposition_diagnostic=True
delta0_eV=0.0
not_final_response_formula=True
not_final_optical_conductivity=True
not_final_Casimir_input=True

## 参数
- kinds=spm, dwave
- omega_list=0, 1e-06, 2e-06, 5e-06, 1e-05, 2e-05, 5e-05, 0.0001
- nk=16
- temperature_K=30
- eta_eV=0.0001

## Delta0=0 在最低 Omega (0 eV) 的比值

ratio 使用 K_total = K_dia - K_para。
- dwave: para_ratio_xx=1+0j, dia_ratio_xx=1+0j, total_ratio_xx=1+0j, para_relative_error=7.75711e-15, dia_relative_error=8.79188e-16, total_relative_error=5.51756e-15
- spm: para_ratio_xx=1+0j, dia_ratio_xx=1+0j, total_ratio_xx=1+0j, para_relative_error=7.75711e-15, dia_relative_error=8.79188e-16, total_relative_error=5.51756e-15

## 最不一致的部分
lowest_omega_mean_para_relative_error=7.75711e-15
lowest_omega_mean_dia_relative_error=8.79188e-16
lowest_omega_mean_total_relative_error=5.51756e-15
largest_lowest_omega_relative_error=para_relative_error

## 下一步
使用该分解判断哪个项需要解析复查。没有单独推导和验证前，不要把这里的
任何 ratio 或符号当作公式修正。

## 图像
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/bdg_vs_normal_K_para_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/bdg_vs_normal_K_dia_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/bdg_vs_normal_K_total_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/para_ratio_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/dia_ratio_xx_vs_omega.png
- validation/outputs/response/bdg_normal_limit_kernel_decomposition/figures/relative_error_vs_omega.png
