# 当前总路线

## 主计算链

```text
H0(k)
→ pairing ansatz
→ finite-q normal/BdG response
→ microscopic point certification
→ reflection/logdet
→ full adaptive outer-Q integration
→ adaptive Matsubara sum
```

外积分工程结构已经闭合：径向、角向、联合误差预算、outer-Q cutoff/tail 和 Matsubara cutoff/tail 均由同一顶层控制器组织。

## 唯一入口

```text
build_full_casimir_config
→ run_full_casimir
```

旧 fixed-grid 路线只位于 `lno327.casimir.legacy`，用于回归比较，不属于主流程。

## 当前物理状态

- 自适应外积分架构完整；
- microscopic 认证、缓存和 fail-closed 传播完整；
- 真实 LNO327 全栈 pilot 尚未成功闭合；
- Ward/gauge 与真实模型资格状态仍决定结果是否可作为物理结论；
- `production_casimir_allowed` 继续为 `false`。

下一步不是增加新的积分维度，而是在一个人为指定的 `spm` 参数点运行正式全栈 pilot，并依据审计证据校准计算上限和成本。
