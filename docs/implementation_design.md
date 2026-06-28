# 代码实现结构

## 设计目标

代码结构的核心原则：

- 理论对象与工程模块尽量一一对应；
- 不确定的模型输入与通用 response engine 分离；
- validation 只诊断，不修正 response；
- raw diagnostic response 不自动进入 Casimir input。

## 仓库顶层分工

- `src/lno327/`：核心计算实现；
- `scripts/`：当前可运行入口；
- `outputs/`：主计算产物和初级结果；
- `validation/`：数值检验、诊断结果、复现命令；
- `docs/`：理论主线和工程设计；
- `docs/references/`：参考文献和背景资料。

## 理论对象到代码模块的映射

| 理论对象 | 代码层 | 说明 |
|---|---|---|
| `H0(k)` / tight-binding representation | `model.py`, `tb_fourier.py` | normal-state model 与 Fourier / hopping 表示 |
| normal response / Kubo baseline | `conductivity.py`, `response_interface.py`, `nonlocal_response.py` | normal-state response 与接口层 |
| pairing ansatz `Delta(k)` | `pairing.py` | 模型输入层与 pairing form factor |
| BdG local response | `bdg_response.py`, `bdg_nonlocal_response.py` | BdG Hamiltonian 和 local / nonlocal response 诊断 |
| finite-q shared primitives | `finite_q_primitives.py` | 通用低层数值工具 |
| generic finite-q engine | `finite_q_engine.py` | 消费 ansatz，不根据 pairing 名称分支 |
| legacy finite-q wrapper | `finite_q_engine.py` | 兼容旧 public API |
| Ward diagnostic | `ward_response.py`, `ward_validation.py` | 只报告 residual，不修 response |
| unit conversion | `response_conventions.py` | response 到 sheet / dimensionless convention |
| reflection input | `reflection_input.py`, `lifshitz_readiness.py` | 下游 input adapter 与 readiness check |
| Casimir scaffold / benchmark | `casimir.py`, `casimir_grid.py`, `casimir_integrand.py`, `casimir_toy_integration.py` | local / prototype Casimir 计算组件 |
| material-grid prototypes | `material_reflection_grid.py`, `material_energy_prototype.py`, `material_grid_convergence.py`, `material_production_grid.py` | material response/reflection/grid planning 候选路径；finite-q material Casimir candidate path 已移除 |
| validation evidence | `validation/` | summary / status / command |
| current material outputs | `outputs/` | 当前主结果 |

## finite-q BdG 分层

`PairingAnsatz` 负责 pairing-dependent 输入：

- `Delta(k)`；
- collective vertices；
- Hubbard-Stratonovich / Goldstone counterterm；
- phase vertex 和 gauge metadata。

generic finite-q engine 只消费 ansatz 和通用数值输入，负责：

- `k-q/2` 与 `k+q/2` 的 BdG eigensystem；
- density / current bubbles；
- direct / contact terms；
- EM-collective mixed kernels；
- collective kernels 和 counterterms；
- Schur-complement responses。

`bdg_finite_q_response_imag_axis` 是兼容 wrapper，用于保留旧 public API。endpoint gauge、`symmetric_kpm`、midpoint 等 phase-vertex 选择属于输入层约定，不属于 generic engine 的 pairing 分支逻辑。

## validation 与 production 的关系

validation 脚本负责检验。validation 输出只长期保存 summary、status 和 command。

Ward validation 不修改 response，不应用 LSQ 或 response-level fitting，不选择新 counterterm，也不把失败结果修成通过。

失败或 diagnostic-only 的 validation 结果不能进入正式 Casimir input。finite-q BdG response 只有在 Ward validation、unit conversion policy 和 `n=0` policy 都明确通过后，才可能成为 formal input。

## outputs 与 docs 的关系

- `outputs/` 保存主计算产物和边界清楚的初级结果；
- `validation/outputs/` 保存验证证据；
- `docs/` 解释理论主线和工程设计，不保存 raw artifact；
- `docs/references/` 保存背景文献，不记录项目进度或 validation 结果。
