# Ward convention / response convention 摘要

## 检验对象

本目录归纳 normal-state response convention 与 Ward residual convention 的审计，包括 Peierls vertex、contact term、density-current response convention、right Ward source convention、equal-time / commutator bookkeeping 和 targeted refinement。

## 已确认内容

- Stage 4.13 支持 positive bubble sign；主路径 bubble prefactor 修正后，total bookkeeping 与 `C-K` 结构一致。
- Stage 4.17 确认 right Ward diagnostic sign convention；旧 right residual 主要是诊断约定问题。
- Stage 4.18 corrected full response Ward validation 达到数值闭合，记录的 `max_corrected_norm` 为 `4.139011615628368e-07`。
- Stage 4.20 targeted clean run 通过 targeted refinement；但 user-run targeted refinement 仍提示部分 cluster 需要更高 refinement 或更宽 Fermi window。
- response-level convention scan 中，physical-current convention with contact minus 是当前最佳 spatial-residual diagnostic case；该结果仍需 analytic convention closure 支撑。

## 当前边界

- 这些检查是 normal-state diagnostic 和 response convention 支撑证据。
- 它们不代表 superconducting finite-q gauge closure。
- 它们不修改 response 公式。
- 它们不使用 LSQ 或 repair 修正 response。
- 它们只支持 response convention，不直接提供 Casimir input。

## 当前结论

Ward / response convention 的 normal-state 约定已得到一组可追踪的数值证据支撑，特别是 corrected right Ward convention 和 targeted refinement 路径。但 finite-q superconducting BdG collective-sector closure 仍由 `bdg_finite_q/` 的 status marker 控制，不能由本目录的 normal-state convention 结果替代。

## 复现入口

主要命令保存在 `command.sh`。运行脚本会重新生成 ignored stage JSON/MD/data/figures。
