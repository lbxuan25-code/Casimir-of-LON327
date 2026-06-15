# Stage 5.3 bilayer sheet conductivity symmetry / convergence audit

## 为什么 Stage 5.2 的 offdiag 不是立刻的错误

Stage 5.2 已经确认 moderate scan 中 Ward residual 闭合、对角 conductivity 为正，并且 Matsubara frequency trend 合理。剩下需要解释的是斜向 finite-\(q\) 下较大的 \(\sigma_{xy},\sigma_{yx}\)。

在 finite-\(q\) response 中，\((x,y)\) 坐标不一定是最自然的张量基底。如果真正自然的响应是 longitudinal / transverse，即 \(L/T\) 基底中的 \(\sigma_{LL}\) 与 \(\sigma_{TT}\)，那么当 \(\mathbf q\) 不是沿坐标轴时，投影回 \((x,y)\) 后会自然产生 symmetric offdiag。这种 offdiag 是张量几何投影，不等同于 Hall response。

## \((x,y)\) 坐标和 \((L/T)\) 坐标

对非零 \(\mathbf q=(q_x,q_y)\)，定义

\[
\hat e_L=\frac{\mathbf q}{|\mathbf q|},\qquad
\hat e_T=(-\hat q_y,\hat q_x).
\]

脚本构造

\[
R=\begin{pmatrix}
\hat q_x & \hat q_y\\
-\hat q_y & \hat q_x
\end{pmatrix},
\qquad
\sigma_{LT}=R\sigma_{xy}R^T.
\]

如果 \((x,y)\) offdiag 较大，但 \((L/T)\) offdiag 显著减小，就支持 finite-\(q\) tensor geometry 的解释。

## symmetric 与 Hall-like antisymmetric offdiag

脚本分解

\[
\sigma^S_{xy}=\frac{\sigma_{xy}+\sigma_{yx}}{2},\qquad
\sigma^A_{xy}=\frac{\sigma_{xy}-\sigma_{yx}}{2}.
\]

若 \(\sigma_{xy}\approx\sigma_{yx}\)，说明 symmetric mixing 占主导，更像有限 \(q\) 下的几何投影。若 \(\sigma_{xy}\approx-\sigma_{yx}\)，才更像 Hall-like antisymmetric response，需要另行审计 TRS breaking 或 source convention。

## q-sign test 为什么重要

比较 \(\mathbf q_+=(q_x,+q_y)\) 与 \(\mathbf q_-=(q_x,-q_y)\)。如果 diagonal 基本相同，而 \(\sigma_{xy},\sigma_{yx}\) 随 \(q_y\to -q_y\) 变号，这说明 offdiag 是有明确空间对称性的 finite-\(q\) 项，不是随机积分噪声。

## q-scaling 与 convergence 为什么必须做

即使 offdiag 能被几何解释，也仍需确认它不是积分误差。Stage 5.3 因此记录 \(q\)-scale trend、adaptive level、Gauss order 和 Fermi window 改变下的相对差异。若 offdiag 对积分参数稳定，且 Ward residual 继续闭合，就更支持它是稳定 finite-\(q\) tensor structure。

## 为什么仍不能进入 reflection/Casimir

Stage 5.3 仍然只是在 model-level bilayer-normalized 2D sheet conductivity 层面做诊断。它没有应用最终 SI sheet scaling，没有构造 reflection matrix，也没有处理 \(n=0\) policy 或 finite-thickness slab 边界条件。因此本阶段最多决定是否进入 Stage 5.4 SI sheet scaling / reflection-input preparation，仍不能声明 Casimir-ready。
