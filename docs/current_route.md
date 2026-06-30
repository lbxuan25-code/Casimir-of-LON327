# 当前总路线

## 总目标

从 LNO327 minimal model 出发，建立 response-to-Casimir 的计算链条：normal / BdG response、finite-q response、Ward / gauge validation、unit / reflection input 与 Casimir benchmark。

## 总路线

```text
H0(k)
-> pairing ansatz
-> BdG / normal response
-> finite-q response
-> Ward / gauge validation
-> unit conversion
-> reflection input
-> Casimir benchmark
```

## 当前主线位置

- local `q=0` response 是 baseline；
- finite-q BdG response engine 是当前 response 主线；
- `PairingAnsatz` 与 generic finite-q engine 的分层是当前工程结构；
- Ward / gauge closure 是 finite-q response 进入 formal conductivity 的关键阻塞；
- unit conversion、reflection input 和 `n=0` policy 是进入 formal Casimir input 的必要 gating；
- 当前 Casimir 相关输出只能作为 benchmark / baseline / candidate，不是最终材料结论。

## 下一步核心问题

- 完成或明确 finite-q response 的 Ward / gauge closure；
- 明确 formal finite-q conductivity 的构造条件；
- 固定 unit conversion 与 reflection input 的 production policy；
- 建立 Matsubara `n=0` policy；
- 在上述 gating 通过后，才讨论正式 Casimir energy / force / torque。

详细数值状态见 `../validation/reports/validation_summary.md`。该 validation 总览是证据入口；本文档只说明项目路线。
