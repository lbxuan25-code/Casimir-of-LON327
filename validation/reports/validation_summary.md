# 当前 validation 状态

## 已在代码 contract 中闭合

| 模块 | 状态 | 说明 |
|---|---|---|
| finite-q microscopic response | implemented | two-band BdG、amplitude/phase Schur、primitive `(A0,Ax,Ay)` |
| finite-q Ward | contract tests pass | arbitrary-q、analytic nonzero RHS、crystal-xy validation |
| positive Matsubara sheet | contract tests pass | `sigma=-K_eff/xi`，仅 `xi>0` |
| exact zero Matsubara | contract tests pass | thermodynamic static Kubo、`chi_bar`、`Dbar_T`、独立静态反射 |
| reflection/logdet | contract tests pass | common lab-LT tangential-E basis、signed real logdet |
| performance workspace | contract tests pass | material/q 两层 cache、cached RHS、batched xi、vectorized contraction |

## 尚待本地数值验证

1. 固定非零 q 的 k-grid 收敛；
2. strict static longitudinal、mixing、reality 和 positivity gates；
3. normal 与 superconducting 的 q→0 transverse 行为；
4. 固定 q 下 exact n=0 与 xi→0+ reflection 匹配；
5. 后续 q、phi、Matsubara quadrature 与 cutoff/tail 收敛。

当前 active 入口是 `python -m validation.run_static_nk_scan`。在上述本地验证完成前，PR 保持 draft，且没有正式 finite-q Casimir 输出。
