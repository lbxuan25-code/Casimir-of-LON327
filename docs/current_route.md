# 当前总路线

## 总目标

本仓库的目标是从 LNO327 minimal model 出发，建立 normal / BdG response、finite-q response、unit / reflection input 与 Casimir benchmark 的计算链条。

当前仓库仍是研究型计算框架，不是最终材料结论仓库。

## 总路线

```text
H0(k)
-> pairing ansatz
-> BdG / normal response
-> finite-q response
-> Ward / gauge consistency validation
-> unit conversion
-> reflection input
-> Casimir observable
```

## 当前进度

当前已经完成或基本完成：

- local `q=0` response baseline 已经形成；
- finite-q BdG q=0 定义对齐已经澄清：`spm` 是 convention-aware pass，`dwave` 是 intraband-aware pass；
- `dwave` raw-vs-total q=0 mismatch 已解释为 local intraband / `-f'(E)` 贡献，不再作为未解释 raw-bubble/vertex mismatch；
- validation 输出已经归档为 summary / status / command；
- finite-q BdG response 已经完成架构解耦；
- generic finite-q engine 与 `PairingAnsatz` 输入层已经分离；
- response、units、reflection 的 validation 证据已经按计算流程整理；
- local-response Casimir benchmark 已形成边界清楚的初级 baseline。

## 当前尚未完成

当前尚未完成：

- finite-q BdG response 的完整 Ward / gauge closure；
- formal finite-q conductivity；
- 可进入正式 Casimir input 的 unit / reflection gated response；
- Matsubara `n=0` policy 的完整处理；
- 最终 Casimir torque、force 或 energy 结论。

## 当前工作重心

当前主线不是直接输出最终 Casimir torque，而是把 response 计算链条整理为可验证、可扩展、边界清楚的结构。

详细数值状态见 `../validation/reports/validation_summary.md`。该 validation 总览是证据入口；本文档只说明项目路线。
