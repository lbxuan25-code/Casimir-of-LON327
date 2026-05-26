# Outputs Guide

本目录保存可复现实验数据、诊断图和接口 smoke 输出。默认原则是：

- `data/` 保存 `.npz` 或 `.csv`，用于复算、重画和表格提取。
- `figures/` 保存 300 dpi `.png`，优先用于论文草稿、组会和笔记。
- `smoke/` 和 `casimir/` 当前只用于接口验证，不代表正式物理结论。

## 论文草稿优先级

优先考虑下列目录中的图和数据作为论文草稿素材：

1. `pairing/gap_structure/`：投影 gap 幅值、near-node 与 preliminary sign 诊断。
2. `normal_state/conductivity_imag/` 和 `normal_state/conductivity_real/`：normal-state Kubo 基线。
3. `bdg/paramagnetic_kernel_imag/`、`bdg/diamagnetic_kernel/`、`bdg/total_kernel_imag/`：
   BdG kernel 层次诊断。
4. `bdg/superconducting_response_imag/`：仅 $n\ge 1$ 的
   $\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega_{\mathrm{eV}}$ 诊断。
5. `response/local_sheet_imag/` 与 `response/static_policy_comparison/`：
   Casimir 前置 response 接口和 $n=0$ policy 边界说明。

## 不应作为论文结论的输出

- `smoke/`：只验证脚本和接口链路。
- `casimir/`：当前预留给未来正式计算。
- `response/static_response/` 和 `response/nonlocal_interface/`：接口边界诊断，不是最终物理方案。
- `response/static_policy_comparison/` 中的 `extrapolate_from_lowest_matsubara` 和
  `use_static_kernel`：只作敏感性或 stiffness-like 静态核诊断。

## 当前 n=0 约定

Lifshitz 求和形式上包含 $n=0$ 半权重项。当前 local isotropic baseline 默认
`n=0 policy = skip`，不是因为 $n=0$ 不存在，而是因为 superconducting
$\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega$ 只定义于 $n\ge 1$。

不要把

$$
K_{\mathrm{total}}(0)/0
$$

定义为 $\Sigma_{\mathrm{SC}}(0)$，也不要把 $K_{\mathrm{total}}(0)$ 直接作为
sheet conductivity 输入 reflection matrix。
