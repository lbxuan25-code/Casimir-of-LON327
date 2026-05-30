# local-response baseline 当前状态

## 已完成内容

response 单位路径已经整理：model response 到 SI sheet conductivity，再到 reflection
dimensionless convention 的入口已经有明确边界。

normal FS-adaptive sampling 已通过数值稳定性检查，当前推荐设置为：

```text
normal_sampling=fs_adaptive
normal_nk=96
normal_refine_factor=8
```

local Casimir integral convergence 已通过 benchmark 级别检查，推荐设置为：

```text
bdg_nk=32
phi_num=32
u_max=80
du=0.5
matsubara_max=64
```

local-response distance scan 已完成。normal、spm、dwave 三个 kind 均保持
zero-torque baseline，未发现 spurious torque。

旧中间诊断已经归档：

- normal sampling 诊断：`outputs/archive/normal_state/`
- local-response convergence / refined convergence：`outputs/archive/casimir/local_response_integral/`
- smoke 输出：`outputs/archive/smoke/`

当前 active local-response 输出保留：

- `outputs/casimir/local_response_integral/distance_scan/`
- `outputs/casimir/local_response_integral/cache/`

注意：cache 仍保留在 active 位置，因为 distance scan command 仍引用该路径。

## 当前用途

local-response 结果现在只作为 baseline：

- 检查数值积分、response cache、distance dependence 和 zero-torque baseline；
- 为 finite-q response 主线提供对照；
- 不作为最终 Casimir torque 结论。

## 边界

- `local_response=True`
- `finite_q_resolved=False`
- `n0_policy=skip`
- `benchmark_only=True`
- `not_final_Casimir_conclusion=True`
