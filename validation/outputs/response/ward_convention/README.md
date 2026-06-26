# Ward / response convention validation

本目录保存 normal-state Ward convention 与 response convention 的轻量验证证据。

## 本目录验证什么

- Peierls current vertex 与 contact term convention；
- density-current Ward residual convention；
- left/right Ward source convention；
- corrected residual diagnostic 是否与当前 response convention 一致。

## 本目录不验证什么

- 不验证 superconducting finite-q BdG gauge closure；
- 不修改 response 公式；
- 不使用 LSQ 或 repair 进入 production pipeline；
- 不提供 Casimir input。

## production relevance

本目录支撑 response convention 和 diagnostic residual convention。它是 production-relevant 的约定证据，但不单独给出 production response。

## diagnostic-only

当前结果是 diagnostic-only；它不替代 BdG finite-q gate。

核心摘要见 `ward_convention_summary.md`。复现入口见 `command.sh`。旧 stage 名称仅在 summary 附录中保留。
