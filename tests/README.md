# tests 入口说明

本目录测试主要验证接口、quick diagnostic 和数值边界，不代表正式物理结论。

## 当前主线测试

- local q=0 response、normal Kubo、BdG response、单位转换和 static policy。
- local-response Casimir benchmark plumbing 和 zero-torque baseline。

local-response distance scan 测试只验证 benchmark plumbing 和 zero-torque baseline，不代表
正式 Casimir torque 结论。

## 历史稳定性测试

normal sampling、high-Nk convergence、BdG response、BdG total kernel、n=0 sensitivity、
static policy、local-response convergence 等测试用于历史稳定性和接口边界复查。

## 测试边界

- 长时间 benchmark 不应放进默认 quick diagnostic 任务。
- 测试通过不代表最终 Casimir input 已经完成。
- 测试通过不代表可以输出正式 Casimir torque 结论。

## pytest markers

`tests/conftest.py` 会按文件名自动加上结构标签，便于把平铺测试临时分层：

- `unit`：基础接口和局部物理恒等式。
- `diagnostic`：研究诊断脚本和 prototype 字段。
- `benchmark`：缩小参数后的 benchmark / convergence plumbing。
- `regression`：历史阶段行为保护。

常用筛选：

```bash
pytest -m unit
pytest -m "diagnostic or regression"
pytest -m "not benchmark"
```
