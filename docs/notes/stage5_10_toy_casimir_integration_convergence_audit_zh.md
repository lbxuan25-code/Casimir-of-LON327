# Stage 5.10 toy Casimir integration convergence audit

## 为什么在真实材料前要做 toy full integration

Stage 5.8 已经验证了单点 trace-log integrand，Stage 5.9 规划了 Matsubara 和 \((Q,\varphi)\) 网格。但真正执行完整求和积分时，还会出现新的数值问题：Matsubara 截断、\(Q_{\max}\)、\(n_Q\)、\(n_\varphi\)、距离依赖和角度依赖都需要单独审计。

Stage 5.10 使用解析 toy reflection matrices 做完整 toy integration，目的是测试积分器和 convergence workflow，而不是计算真实材料能量。

## toy integration 与真实 Casimir energy 的区别

本阶段计算的是

\[
\frac{\mathcal F_{\rm toy}}{A}
=
k_BT\sum_n'
\int_0^{Q_{\max}}
\frac{Q\,dQ}{(2\pi)^2}
\int_0^{2\pi}d\varphi\,
\log\det\left[I-e^{-2\kappa d}R_1^{toy}R_2^{toy}\right].
\]

这里的 \(R^{toy}\) 是人为构造的解析矩阵，不来自 LNO327 response。输出只能解释为 toy-model integration audit，不能解释为 LNO327 Casimir energy、force 或 torque。

## 为什么需要收敛审计

正式计算需要检查：

- \(n_{\max}\)：Matsubara 高频截断；
- \(Q_{\max}\)：大 \(Q\) 截断；
- \(n_Q\)：径向积分分辨率；
- \(n_\varphi\)：角向积分分辨率；
- \(d\)：距离依赖是否数值稳定；
- \(\theta\)：角度依赖是否来自模型而不是网格噪声。

Stage 5.10 的 scan 很轻量，只用于建立流程，不是 production-quality convergence。

## 为什么 toy model 使用 cutoff

toy reflector 使用

\[
f(\xi,Q)=
\exp[-(Q/Q_c)^2]\frac{1}{1+\xi/\xi_c}.
\]

这个 cutoff 让大 \(Q\) 和高频贡献自然衰减，避免无限积分在 toy audit 中变成数值病态问题。

## isotropic toy 不应有角度依赖

isotropic toy reflector 是 diagonal 且只依赖 \(\xi,Q\)，不依赖 \(\varphi\) 或 \(\theta\)。因此

\[
\mathcal F_{\rm toy}^{iso}(\theta)
\]

应在数值误差内不随 \(\theta\) 变化。

## anisotropic toy 的角度依赖含义

anisotropic symmetric toy reflector 使用

\[
R_2^{toy}(\theta)=O(\theta)R_0^{toy}O(\theta)^T.
\]

这个 rotation 只用于测试积分器对角度变化的响应，并不代表真实材料旋转。真实材料旋转仍需要通过

\[
Q^{crystal}=R(-\theta)Q^{lab}
\]

重新查询 response，再把最终 \(R^{TE/TM}\) 表达到共同 lab basis。

因此 toy anisotropic angle dependence 不能解释为 LNO327 torque。

## 下一步真实材料需要什么

进入真实材料前需要 production grid：

\[
R^{TE/TM}(i\xi_n,Q,\varphi)
\]

或

\[
\tilde\sigma(i\xi_n,Q,\varphi)
\]

并完成 response-grid convergence audit、插值策略审计、\(Q=0\) 处理、高频和大 \(Q\) 截断审计。之后才适合考虑真实材料 energy 或 torque。
