# Unit conversion / reflection input 摘要

## 检验对象

本目录归纳 response-to-sheet-conductivity、model units to SI / dimensionless sheet conductivity、reflection input tensor formatting、TE/TM reflection adapter，以及 q-grid / model-q mapping 相关诊断。

## 当前状态

- Stage 5.1 初始 convention audit 指出 unit chain ambiguous，不能直接进入反射或 Casimir。
- Stage 5.1b 固定 bilayer sheet model convention。
- Stage 5.2 / 5.3 是 conductivity sanity 和 symmetry / convergence monitor，其中 Stage 5.3 仍要求 source symmetry 进一步审计。
- Stage 5.3b、5.4a、5.4b 通过了 finite-q lattice tensor effect、unit conversion 和 dimensionless sheet conductivity 的候选检查。
- Stage 5.5b reflection input tensor formatter 通过。
- Stage 5.6 TE/TM adapter 通过。
- Stage 5.8-5.13 是 scaffold/prototype/convergence planning 路径，不代表完整 Lifshitz/Casimir pipeline。
- material reflection grid 有早期失败和后续 monitor / passed 版本；这些仍是 diagnostic candidate，不是 production result。

## 明确边界

本目录不计算正式 energy、force 或 torque。没有 Ward validation、unit policy、n=0 policy 同时闭合时，raw response 不能进入正式 Casimir input。

Stage 5 prototype / scaffold 的通过只说明格式、单位链、adapter 或 toy integration 的局部检查可运行；它不提升 raw finite-q BdG response 为 production-ready。

## 复现入口

主要命令保存在 `command.sh`。运行脚本会重新生成 ignored stage JSON/MD 和其他 artifact。
