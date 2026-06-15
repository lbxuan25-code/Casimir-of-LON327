# Stage 5.3b finite-q offdiag 收敛性审计

## 为什么 Stage 5.3 还不足以最终判断

Stage 5.3 已经表明，轴向 \(q_x,q_y\) 的 offdiag 基本为零，而斜向 \(q_{\rm diag\pm}\) 的 offdiag 明显；并且 \(q_y\to -q_y\) 时 offdiag 变号，\(\sigma_{xy}\approx\sigma_{yx}\)。这些都支持 finite-q tensor structure，而不是 Hall-like response。

但 Stage 5.3 的主要 moderate 结果只使用了一个积分配置，因此还不能排除 offdiag 来自 adaptive level 或 Fermi window 的积分误差。Stage 5.3b 只针对最关键的低频 diagonal-q offdiag 做 targeted convergence audit。

## 为什么只看 low Matsubara

Stage 5.2 中 offdiag 在 \(n=1,2\) 最大，随 Matsubara frequency 升高下降。因此最需要收敛性确认的是低频点。若低频 offdiag 在不同积分参数下稳定，高频通常更不容易成为 blocker。

## 为什么 q-sign 和 q-scaling 关键

\(q_{\rm diag+}\) 与 \(q_{\rm diag-}\) 的 diagonal even、offdiag odd 结构说明该项有明确空间对称性。比较 \(q\) 与 \(0.5q\) 则检查 finite-q origin：若 \(0.5q\) 下 offdiag 下降，说明它不是 \(q\to0\) 后仍残留的坐标或 source bug。

## symmetric 与 Hall-like antisymmetric offdiag

若 \(\sigma_{xy}\approx\sigma_{yx}\)，offdiag 是 symmetric mixing；若 \(\sigma_{xy}\approx-\sigma_{yx}\)，才更接近 Hall-like antisymmetric response。Stage 5.3b 因此持续记录 \(|\sigma^A_{xy}|/|\sigma^S_{xy}|\)。

## 后续如何处理稳定 offdiag

如果 Stage 5.3b 证明 offdiag 对 level/window 稳定、Ward 闭合、q-sign 和 q-scaling 均稳定，则后续应把它描述为 stable finite-q lattice tensor effect。即使 simple continuum \(L/T\) 投影不能完全消掉它，也不自动说明错误；这可能说明 lattice finite-q tensor structure 需要更完整的 projector/source decomposition。

## 为什么仍不能进入 Casimir

本阶段仍是 model-level bilayer-normalized sheet conductivity 诊断，没有做 SI sheet scaling、reflection input preparation、\(n=0\) policy 或 finite-thickness slab 边界条件。因此 Stage 5.3b 通过后也只能进入 Stage 5.4 准备阶段，不能直接声明 Casimir-ready。
