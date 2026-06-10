# 数值稳定性阶段总结

## 1. 本阶段目标

本阶段目标是归纳并固定当前数值稳定性证据，确认 response 层和 local-response
Casimir integral benchmark 可以进入下一轮距离扫描基准测试。

本阶段只验证数值稳定性，不输出正式 Casimir 结论。当前所有 Casimir 相关输出仍是
benchmark-only：用于检查积分、采样、截断和基线是否稳定，而不是用于声明真实
Casimir torque 的大小、符号或可观测结论。

## 2. 已完成的数值稳定性测试

### 正常态响应的 k-space / FS-adaptive sampling 收敛

normal-state low-Matsubara response 已完成 uniform、multishift_average 与
fs_adaptive 的采样比较。最终窄范围复查显示，`fs_adaptive` 在低 Matsubara 点上满足
`Nk=96 -> 112` 的 2% 阈值要求，并且 `refine_factor=8 -> 10` 的变化很小。

因此当前 normal response 推荐使用 FS-adaptive 采样作为 local-response benchmark 的
normal 输入。

### BdG 响应的对称性、Delta0->0、eta/Nk 稳定性

BdG response 已完成基础对称性诊断、`Delta0 -> 0` 的 BdG-normal 极限检查，以及
imaginary-axis response 对 `eta`、`Nk` 和 Matsubara index 的稳定性检查。

这些测试的用途是确认 response 层没有明显发散、不连续或对称性破坏。它们不构成
正式 optical conductivity 或正式 Casimir 输入的最终验证。

BdG response contract 已更新：`K_para` 表示正的 current-current bubble，保留
BdG particle-hole redundancy 的 Nambu `1/2` prefactor；`K_dia` 表示 mass/contact
term。根据 normal-state Peierls/free-energy stiffness 诊断，当前 `K_total` 的
含义为 `K_dia - K_para`，不再使用旧加法 convention。由此得到的
`Sigma_SC=K_total/omega` 仍只是 positive Matsubara 上的 sigma-like response
diagnostic，不是最终 optical conductivity 或最终 Casimir 输入。

### 局域响应 Casimir 积分的基础收敛

local-response Casimir integral 已完成基础 Matsubara、`k_parallel` cutoff/grid 和
`phi` grid 收敛检查。基础收敛暴露出两个需要进一步处理的 blocker：
旧 cutoff scan 中固定 `kparallel_num` 会导致 cutoff 增大时 `du` 同时改变；
Matsubara tail 需要更长截断复查。

### 细化 cutoff 扫描

refined cutoff scan 已改用
`u = k_parallel * distance`
并保持固定 `du`。在 `u_max=20, 40, 60, 80`、`du=0.5` 的设置下，normal、spm、
dwave 三个 kind 均达到 clean cutoff convergence。

### 扩展 Matsubara 扫描

extended Matsubara scan 已覆盖 `matsubara_max=24, 32, 48, 64`。在最大截断
`matsubara_max=64` 处，normal 为 `candidate_converged`，spm 与 dwave 为
`loose_converged`。这些设置已经用于主 `outputs/` 中的 local-response distance scan
初级结论，但仍不是最终 Casimir 结论。

### 零力矩 baseline 检查

当前 refined benchmark 的 normal、spm、dwave 均通过 zero-torque baseline 检查，
未发现超过阈值的 spurious torque。该结论只说明当前 local isotropic benchmark
没有暴露出数值伪力矩，不说明真实物理 torque 必然为零。

## 3. 关键结果

normal response 推荐设置：

```text
normal_sampling=fs_adaptive
normal_nk=96
normal_refine_factor=8
```

local integral 推荐设置：

```text
bdg_nk=32
phi_num=32
u_max=80
du=0.5
matsubara_max=64
```

当前 refined local-response convergence 的关键状态：

- clean cutoff scan 已通过；
- Matsubara normal 为 `candidate_converged`，spm/dwave 为 `loose_converged`；
- zero-torque baseline 成立；
- 没有发现 spurious torque；
- 当前已完成 local-response distance scan 初级结论。

finite-q Stage 1 已重启为 normal current-current kernel convergence diagnostic，
并作为 validation/response 输出保留。它只验证
$K(i\omega_n,\mathbf{q})\to K(i\omega_n,\mathbf{0})$，不属于 local-response
Casimir baseline，也不是 finite-q conductivity 或 Casimir 输入。后续 finite-q
路线见 `docs/notes/casimir_torque_response_pipeline_zh.md`。

## 4. 为什么仍然不能称为正式 Casimir 结论

当前阶段仍保留以下边界：

```text
local_response=True
n0_policy=skip
benchmark_only=True
not_final_casimir_conclusion=True
```

这意味着当前结果只覆盖 local `q=0` response 的 benchmark 积分，尚未包含有限动量
Lifshitz response。`n=0` Matsubara 项仍采用保守的 `skip` policy，不能把
当前结果解释为包含完整零频反射模型的正式 Casimir 计算。

此外，当前 benchmark 尚未引入真实 torque 来源机制。zero-torque baseline 的成立只
说明当前各向同性 local benchmark 没有数值伪力矩；它不是“真实无效应”的物理结论。

## 5. 当前主结果与下一阶段

local-response distance scan 已从数值稳定性任务提升为边界清楚的初级结论，保存在：

```text
outputs/casimir/local_response_distance_scan/
```

该结果保持 `local_response=True`、`finite_momentum_resolved=False`、
`n0_policy=skip`、`benchmark_only=True` 和
`preliminary_local_response_conclusion=True`。下一阶段应优先处理 n=0 policy 与
真实 torque 来源机制。

## 6. 当前不允许做的事

- 不允许写正式 Casimir torque 结论；
- 不允许把 local benchmark 当有限动量 Lifshitz 结果；
- 不允许把 zero-torque baseline 解释成真实无效应结论；
- 不允许改变当前 `n0_policy=skip` 边界后直接复用本阶段结论；
- 不允许在尚未引入真实 torque 来源机制前声称已经得到物理 torque。

下一阶段应围绕 n=0 policy、真实 torque 来源机制、local Casimir 初级结论的适用
边界，以及当前总览路线图中的 finite-q 单位审计和 Ward convention 闭合展开。
