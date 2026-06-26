# 理论计算路径

## 物理目标

本仓库关注从 LNO327 minimal model 的 normal / superconducting response 出发，构造未来可能用于 reflection / Casimir benchmark 的响应输入。

本文档描述理论路径，不宣称当前已经得到最终 Casimir torque、force 或 energy 结论。

## 模型 Hamiltonian

normal-state Hamiltonian 是计算链条的起点。当前 minimal model 使用四轨道基

```text
(dz1, dx1, dz2, dx2)
```

其中 `dz` 表示 `d_z^2` 轨道，`dx` 表示 `d_{x^2-y^2}` 轨道，`1/2` 表示双层自由度。normal Hamiltonian 可写成双层 block 结构

```text
H0(k) = [[H_parallel(k), H_perp(k)],
         [H_perp(k),    H_parallel(k)]] - mu I.
```

该 Hamiltonian 提供能带、本征态、velocity / mass vertices 和后续 BdG 构造的 normal-sector 输入。

## 配对 ansatz

pairing ansatz 是 BdG response 的模型输入，不是 generic response engine 的一部分。当前 minimal ansatz 主要包括：

- `s_pm`：层间 `d_z^2` 结构；
- `dwave` / `B1g`：同层 `d_z^2`-`d_{x^2-y^2}` interorbital 结构；
- `Delta(k)`：进入 BdG Hamiltonian 和 collective-channel vertices 的 pairing matrix。

pairing ansatz 决定 form factor、phase vertex、collective vertices 和相关 counterterm metadata。它不应让 generic finite-q engine 根据 pairing 名称分支。

## BdG Hamiltonian 与 local response

BdG Hamiltonian 采用标准 Nambu block：

```text
H_BdG(k) = [[ H0(k),        Delta(k)],
            [ Delta†(k),   -H0^T(-k)]]
```

current vertex 和 contact / mass vertex 由 normal-sector Hamiltonian 的一阶、二阶动量导数嵌入 Nambu 空间。local `q=0` response 主要用于建立当前 baseline，并区分 paramagnetic kernel、diamagnetic / contact contribution 和 total kernel convention。

当前 positive Matsubara 上的 sigma-like superconducting response 只作为 imaginary-axis diagnostic。它不是实频 optical conductivity，也不自动成为 Casimir input。

## finite-q response

Casimir / reflection 问题需要 finite in-plane momentum `q`，因为外部电磁场或真空涨落会携带材料表面的平行动量。

local `q=0` conductivity 不能直接代表 finite-q response。finite-q response 至少需要区分：

- density sector；
- current sector；
- contact sector；
- external bosonic Matsubara frequency；
- external in-plane momentum `q`。

bare current-current kernel 只能作为 diagnostic object；它不等于 gauge-closed finite-q conductivity。

## Ward / gauge consistency

Ward identity 的理论作用是约束 density-current-current response，并检查 gauge consistency。

需要明确：

- Ward residual 小是必要诊断，但 residual 最小不是物理推导；
- response-level fitting 或 LSQ 不能替代 Ward closure；
- contact sign、current sign、density vertex、left/right convention 必须在同一约定下使用；
- 未闭合的 finite-q response 不能作为 formal reflection / Casimir input。

## conductivity / sheet response

response 到 conductivity / sheet conductivity 的转换是下游步骤，需要清楚单位和 convention。

裸 kernel 不能直接称为 optical conductivity。model response 到 SI sheet conductivity，再到 dimensionless sheet conductivity 的路径必须在 unit policy 中说明，并且不能重复乘单位转换因子。

## reflection input

reflection matrix 需要 unit-converted、validated response input。reflection adapter 的职责是把 response / conductivity 放入 reflection matrix 格式。

adapter 本身不计算 Lifshitz energy、force 或 torque，也不能绕过 upstream Ward validation、unit conversion policy 或 `n=0` policy。

## Casimir observable

理论上最终目标是通过 validated reflection input 进入 Lifshitz / Casimir observable，例如 energy、force 或 torque。

当前仓库尚未给出最终 Casimir torque、force 或 energy 结论。当前 local-response benchmark 只能作为边界清楚的初级 baseline；finite-q formal Casimir input 仍被 Ward / gauge closure、unit policy 和 `n=0` policy 阻塞。
