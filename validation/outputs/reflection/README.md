# Reflection validation outputs

本目录保存 reflection 层级 validation 的轻量证据。报告按具体检验对象组织，重点说明 reflection input 判据、当前状态和复现入口。

## 子目录

- `reflection_input/`：reflection input tensor formatting、TE/TM adapter、pre-Lifshitz readiness 和 prototype reflection/grid 状态。

## 边界

本目录不验证 response Ward closure，不验证 conductivity unit policy 本身，也不计算正式 Casimir energy、force 或 torque。需要复查 raw artifacts 时运行对应 `command.sh`。
