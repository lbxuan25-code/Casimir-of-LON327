# 验证协议

当前 sandbox 只给出诊断协议，不给出生产科学结论。

## Matsubara 频率接口

公开扫描接口使用 Matsubara index `n` 与 `temperature_K`，内部响应频率为

```text
xi_eV = 2*pi*n*k_B*T
```

response layer 计算 `K_TMTE(q, xi_eV)`。`n=0` 时 `xi_eV=0.0`，但未来 Casimir Matsubara 求和中的零模半权重属于 Casimir summation layer，不属于本 sandbox response layer。

## 直接 G/TM/TE gauge 诊断

主输出是 Schur 后的 `K_GTMTE_eff` 与切出的 `K_TMTE_eff`。在诊断源顺序 `["G", "TM", "TE"]` 下记录：

```text
gauge_row_norm = norm(K_eff[G, :])
gauge_col_norm = norm(K_eff[:, G])
gauge_gg_norm  = abs(K_eff[G, G])
```

这些量只衡量目标基中纯规范源的残差，不构成 Ward repair。

sandbox v1 的主诊断路径只接受 `["G", "TM", "TE"]`，不能把 physical-only `["TM", "TE"]` 静默送入 gauge diagnostics。

## Shifted mesh 稳定性

若使用 shifted mesh，必须先平均 bare blocks，再 Schur。记录每个 shift 的 block norm 与条件数，并记录主结果的平均顺序：

```text
average_order = "average_blocks_then_schur"
```

平均 per-shift Schur 仅可作为 debug 参考。

## JSON 结果语义

`tmte_scan.json` 顶层只表示扫描级信息，并包含 `frequency` metadata：`source="matsubara_index"`、`matsubara_index`、`temperature_K`、`xi_eV` 和 `zero_matsubara_mode`。完整的 `K_GTMTE_eff`、`K_TMTE_eff`、bare blocks 和诊断量必须从 `results[i]` 读取。

`first_result_summary` 只用于快速预览第一个 q/方向的轻量诊断，不是全局响应，也不包含完整矩阵。

若某个结果的 Schur solve method 为 `pinv_diagnostic`，JSON 会同时给出 `numerically_suspect=true`。该结果可保留为诊断对象，但需要审阅 collective kernel 条件数与稳定性。

## nk 稳定性

人工后续可对多个 `nk` 做小范围稳定性比较，但不能在实现阶段运行昂贵扫描，也不能仅凭 sandbox v1 输出声明物理收敛。

## 可选 component reference

`debug_compare_component_reference.py` 可以把直接目标基结果与完整分量响应的事后收缩比较。该脚本只用于 debug，不属于主计算路径，不能作为生产 TM/TE 路径。v1 只支持 `q=(q_value, 0)`，不能作为一般方向验证证据。

## Casimir-readiness 限制

在 Casimir kernel 归一化、单位转换、反射输入约定和 Ward/数值稳定性全部推导并验证前，输出必须保持：

```text
valid_for_casimir_input = false
```
