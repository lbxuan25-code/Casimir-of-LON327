# tests 入口说明

本目录测试主要验证接口、quick diagnostic 和数值边界，不代表正式物理结论。

## 当前主线测试

- `test_finite_q_raw_q0_consistency.py`
- `test_casimir_local_response_distance_scan.py`

旧 finite-q 测试仍保留用于回归，脚本路径已指向 `scripts/archive/finite_q_diagnostics/`。
finite-q 测试只验证 response 层 prototype 的字段、flags、quick 运行和诊断输出。
local-response distance scan 测试只验证 benchmark plumbing 和 zero-torque baseline，不代表
正式 Casimir torque 结论。

## 历史稳定性测试

normal sampling、high-Nk convergence、BdG response、BdG total kernel、n=0 sensitivity、
static policy、local-response convergence 等测试用于历史稳定性和接口边界复查。

## 测试边界

- 长时间 benchmark 不应放进默认 quick diagnostic 任务。
- 测试通过不代表 Ward 完备 finite-q response。
- 测试通过不代表最终 Casimir input 已经完成。
- 测试通过不代表可以输出正式 Casimir torque 结论。
