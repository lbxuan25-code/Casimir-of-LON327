# reflection input validation

本目录验证 reflection input formatting 和 TE/TM adapter 的轻量证据。

## 本目录验证什么

- dimensionless sheet conductivity 到 reflection input tensor 的 formatting；
- TE/TM reflection adapter convention；
- pre-Lifshitz readiness；
- prototype / scaffold 层面的 reflection grid 与 toy integration 状态。

## 本目录不验证什么

- 不验证 raw response 的 Ward closure；
- 不验证 conductivity unit policy 是否最终接受；
- 不验证完整 Lifshitz / Casimir production pipeline；
- 不计算正式 energy、force 或 torque。

## production relevance

该目录对 reflection input formatting 有支撑意义，但必须等待 response gate、unit policy 和 `n=0` policy 同时接受后，才可能进入 production。

## diagnostic-only

当前状态为 diagnostic-only / candidate。prototype / scaffold 通过只说明局部格式和 adapter 检查可运行，不能提升 raw finite-q BdG response。

摘要见 `reflection_input_summary.md`，状态见 `reflection_input_status.json`，复现入口见 `command.sh`。
