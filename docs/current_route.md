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

## 临时 best-effort plumbing lane

为了先跑通 finite-q response 到 unit conversion、reflection adapter 和 Casimir-grid plumbing 的下游接口，可以暂时启用 `docs/best_effort_finite_q_casimir_route.md` 中定义的 best-effort finite-q 路线。

这条路线采用当前 Ward-refinement 诊断中最高一档的 adaptive Fermi-window quadrature 参数作为数值默认值，但必须保持：

```text
diagnostic_only=True
best_effort_plumbing=True
ward_identity_closed=False
valid_for_casimir_input=False
not_final_casimir_conclusion=True
```

它的目标是暴露下游接口、单位、reflection、grid/cache 和积分问题；它不改变正式 gating，也不允许输出正式 Casimir energy / force / torque 或材料结论。

## 下一步核心问题

- 完成或明确 finite-q response 的 Ward / gauge closure；
- 明确 formal finite-q conductivity 的构造条件；
- 固定 unit conversion 与 reflection input 的 production policy；
- 建立 Matsubara `n=0` policy；
- 在上述 gating 通过后，才讨论正式 Casimir energy / force / torque。

详细数值状态见 `../validation/reports/validation_summary.md`。该 validation 总览是证据入口；本文档只说明项目路线。
