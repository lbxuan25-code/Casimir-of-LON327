# 输出目录

`outputs/` 只保存当前主计算产物的轻量说明、summary、必要图表和复现入口。

## 当前边界

- `normal_state/`、`pairing/`、`bdg/`：历史主计算的轻量说明；
- `casimir/local_response_distance_scan/`：明确标注为 local-response baseline / benchmark，不是 finite-q production 结果；
- 当前尚无正式 finite-q Casimir energy、force 或 torque 输出。

旧 `finite_q_bdg_pipeline` 产物已经删除，因为对应实现使用过失效的 Ward、零模外推和 TE/TM-amplitude 路线。

## 维护原则

- 正式主流程结果进入 `outputs/`；
- 数值验证和 benchmark 进入 `validation/outputs/`；
- raw 数组、cache、scratch log 和大型表格不提交；
- 不建立第三套根目录 `results/`。
