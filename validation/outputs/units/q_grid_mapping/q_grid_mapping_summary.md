# q-grid mapping 摘要

## 检验对象

本目录归纳 q-grid / model-q mapping 与 Casimir grid planning scaffold 的诊断。它关注积分网格、model-q 覆盖和 warning 是否被明确记录。

## 当前状态

- Stage 5.9 grid scaffold：`STAGE5_9_CASIMIR_GRID_SCAFFOLD_PASSED`。
- scaffold 明确 warning：`Q=0` 的 TE/TM in-plane direction 需要 symmetry/limit 处理或从 angular-grid production runs 排除。
- scaffold 明确 warning：既有 8 个 validation reflection cases 不是 production integration grid。
- historical q-grid model audit 是 unit/sampling audit，不计算 response tensor，不产生 finite-q conductivity，不给出 Casimir 结论。
- historical q-grid model audit 的 full-grid `q_model_max = 1.05333`，`q_model_max/pi = 0.335286`。
- Stage 1 sampled q range 最大为 `0.005`，因此只测试 small-q limit，不覆盖当前 Casimir-relevant q_model range。
- recommended q-list 被分为 small-q regression list、Casimir-relevant q list 和 BZ stress list；BZ stress list 只用于数值压力测试。

## 结论

q-grid / model-q mapping 有可复查的 planning 证据，但不代表 production integration grid 已完成，也不代表正式 Casimir pipeline 已验证。

## 边界

本目录不验证 response、reflection adapter 或 energy integration。当前结果为 diagnostic-only。
