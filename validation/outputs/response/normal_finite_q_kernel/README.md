# normal finite-q kernel

本目录验证 normal-state finite-q current-current kernel 的数值一致性。

## 本目录验证什么

- `q=0` 和 `q!=0` 是否使用同一 finite-q kernel 接口；
- kernel components 是否 finite；
- 小 q same-interface error 和 C4 covariance error 是否处于可追踪范围。

## 本目录不验证什么

- 不验证完整 finite-q conductivity；
- 不验证 gauge-closed response；
- 不处理 `n=0` true static；
- 不提供 reflection 或 Casimir input。

## production relevance

该目录对 normal-state kernel 接口一致性有支撑意义，可作为 response pipeline 的低层数值证据。

## diagnostic-only

该结果仍是 kernel-level diagnostic。它不直接进入 production Casimir pipeline。

核心摘要见 `normal_finite_q_kernel_summary.md`。复现入口见 `command.sh`。
