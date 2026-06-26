# conductivity conversion validation

本目录验证 response 到 conductivity 的单位和规范转换链。

## 本目录验证什么

- response-to-sheet-conductivity convention；
- bilayer sheet model normalization；
- model units 到 SI sheet conductivity；
- SI sheet conductivity 到 dimensionless sheet conductivity。

## 本目录不验证什么

- 不验证 Ward identity；
- 不验证 reflection matrix 公式；
- 不验证完整 Lifshitz / Casimir pipeline；
- 不计算正式 energy、force 或 torque。

## production relevance

该目录对单位链和 conductivity formatting 有支撑意义；但只有在 Ward validation、unit policy 和 `n=0` policy 同时接受后，才可能成为 production pipeline 的一部分。

## diagnostic-only

当前状态为 diagnostic-only / candidate。旧 stage 名称只在 summary 的历史来源对照中保留。

摘要见 `conductivity_conversion_summary.md`，状态见 `conductivity_conversion_status.json`，复现入口见 `command.sh`。
