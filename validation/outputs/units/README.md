# Units validation outputs

本目录保存单位相关 validation 的轻量证据。报告按具体检验对象组织，重点说明单位链判据、当前状态和复现入口。

## 子目录

- `conductivity_conversion/`：response 到 sheet conductivity、SI sheet conductivity、dimensionless sheet conductivity 的单位链检查。
- `q_grid_mapping/`：Casimir / reflection planning 所需 q-grid 与 model-q mapping 检查。

## 边界

本目录不验证 Ward closure，不验证 finite-q BdG superconducting response，也不计算正式 Casimir energy、force 或 torque。需要复查 raw artifacts 时运行对应 `command.sh`。
