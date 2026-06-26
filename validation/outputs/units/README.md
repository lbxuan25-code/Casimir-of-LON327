# Units validation outputs

本目录保存单位相关 validation 的轻量证据，不保存 raw numerical artifacts。

## 子目录

- `conductivity_conversion/`：response 到 sheet conductivity、SI sheet conductivity、dimensionless sheet conductivity 的单位链检查。
- `q_grid_mapping/`：Casimir / reflection 积分所需 q-grid 与 model-q mapping 的诊断。

本目录不验证 Ward closure，不验证 finite-q BdG superconducting response，也不计算正式 Casimir energy、force 或 torque。
