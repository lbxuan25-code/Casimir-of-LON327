# finite-q 诊断流水线

finite-q response 目前只用于诊断 Ward closure。它不是 Casimir-ready 输入，也不应被传入正式 Casimir 公式。

## 模块边界

- `model.py` 负责 normal-state Hamiltonian 和 normal-state electromagnetic vertices。
- `pairing_ansatz.py` 负责 pairing 输入：mean pairing、collective vertices、phase-vertex 约定和当前 gap-equation counterterm provider。
- `finite_q_engine.py` 是通用 finite-q response calculator。它消费 `PairingAnsatz`，不根据 pairing 名称分支。
- `finite_q_diagnostics.py` 是基础诊断 workflow。它显式构造 ansatz，计算 `bare_total`、`minus_schur`、`amplitude_phase_schur`，然后对这些矩阵做 Ward 检查。
- `q0_bdg_response_alignment.py` 检查 q=0 时 finite-q BdG 定义和既有 local response 定义是否对齐。
- `finite_q_ward_scan.py` 扫描小 q 下的 Ward 残差。
- `dwave_pairing_tangent_diagnostics.py` 检查 `dwave` bond/orbital 重构和 endpoint-gauge tangent。
- `goldstone_counterterm_diagnostics.py` 检查 Goldstone counterterm 与 `eta2 = delta0 * theta` 归一化。
- `ward_validation.py` 只检查 Ward residuals，不修补、不改写 response。
- `casimir.py` 保持分离，不能消费 finite-q 诊断 response。

## 新诊断默认值

新的 finite-q 诊断应显式指定：

```python
phase_vertex = "bond_endpoint_gauge"
current_vertex = "peierls"
collective_mode = "amplitude_phase"
collective_counterterm = "goldstone_gap_equation"
include_phase_phase_direct = True
```

legacy finite-q wrapper 会保留历史默认值以兼容旧 public API。因此，新诊断代码不应依赖 wrapper 默认值。

## Casimir gating

所有 finite-q 诊断报告都必须保持：

```python
valid_for_casimir_input = False
```

finite-q response 尚未升级为 production Casimir pipeline，也没有完成 unit conversion、reflection input 和 Ward closure 认证。诊断失败应被报告为失败，而不是通过 response-level 拟合、残差投影或修复来隐藏。
