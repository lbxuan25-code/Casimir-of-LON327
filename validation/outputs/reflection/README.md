# Reflection validation outputs

本目录保存 reflection 层级 validation 的轻量证据。报告按具体检验对象组织，旧 stage 名称只在历史来源对照中保留。

## 子目录

- `reflection_input/`：reflection input tensor formatting、TE/TM adapter、pre-Lifshitz readiness 和 prototype reflection/grid 状态。

## 边界

本目录不验证 response Ward closure，不验证 conductivity unit policy 本身，也不计算正式 Casimir energy、force 或 torque。raw artifacts 已清理，需要复查时运行对应 `command.sh`。
