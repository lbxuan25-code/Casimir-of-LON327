# 迁移计划

任何代码从 sandbox 迁入 `src` 前，至少需要满足：

1. 直接目标基公式通过理论层单元测试和 component debug reference。
2. shifted mesh 的主路径确认是先平均 bare blocks 后 Schur。
3. `K_TMTE_eff` 的模型单位、SI 转换和 Casimir kernel 归一化已单独推导。
4. 纯规范 `G` 残差在目标验证网格上有明确收敛行为。
5. `nk`、`q`、`xi`、pairing channel 的稳定性范围已经记录。
6. 现有生产 Casimir pipeline 有明确接口变更设计和回归测试。
7. 旧 component-basis 诊断仍可作为 debug 参考，但不会被误认为新主路径。

迁移时应先迁入纯理论工具和数据 schema，再迁 adapter，最后才考虑生产 pipeline 接线。

