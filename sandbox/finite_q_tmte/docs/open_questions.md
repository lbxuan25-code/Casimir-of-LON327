# 未决问题

- Casimir kernel normalization 尚未推导完成。`K_TMTE_eff` 目前只是模型单位中的目标基响应候选。
- 目标基响应到 sheet conductivity/reflection input 的单位映射尚未建立。
- 当前 contact 实现只使用空间 `Dxx/Dxy/Dyx/Dyy` 路径；若未来现有代码提供更一般的 component contact，需要重新审计是否存在 `A0` 或混合 contact 项。
- 行侧 observable 与列侧 source 在现有 BdG 后端中包含电流符号约定；sandbox adapter 已隔离该约定，但仍需物理审阅。
- collective amplitude/phase normalization 沿用现有 finite-q BdG 后端；是否正好匹配目标 TM/TE Casimir kernel 仍未证明。
- `q=0` 在本路径中被显式禁止；零模或小 q 外推需要单独设计。
- shifted mesh 与 endpoint/contact quadrature 的最佳组合仍是验证问题，v1 不引入 `ward_conservative` 或 Ward repair。
- debug component reference 目前只支持 q 沿 x 方向，未来若要用于一般方向排查，需要单独扩展和测试。
- 出现 `pinv_diagnostic` 的 Schur 结果需要单独的 collective-kernel 稳定性判据；v1 只把它标记为 numerically suspect。
- Matsubara `n=0` 的 Casimir 半权重、零模电磁边界条件和 q 外推策略属于未来 Casimir layer；本 sandbox 只输出 `K_TMTE(q, xi_eV=0)` 诊断对象。
