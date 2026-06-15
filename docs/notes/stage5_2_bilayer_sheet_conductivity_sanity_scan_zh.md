# Stage 5.2 bilayer sheet conductivity sanity scan

## 为什么做 Stage 5.2

Stage 4.20 已经确认 finite-q physical response 的 Ward residual 在可靠积分参数下闭合。Stage 5.1b 又固定了从 response spatial block 到 model-level bilayer sheet conductivity 的约定：

\[
\sigma^{\rm model}_{ij}(i\Omega)
=-\frac{\Pi_{ij}(i\Omega)}{\Omega_{\rm eV}},
\qquad
\Pi_{ij}=response[1:3,1:3].
\]

Stage 5.2 的目的不是进入 reflection 或 Casimir，而是先检查这个 \(\Pi\to\sigma\) 输出是否具备最基本的数值行为：对角元符号、非对角元大小、Matsubara frequency 平滑性、积分参数稳定性，以及每个 conductivity 点对应的 Ward residual。

## 对角 conductivity 为什么应为正

在 imaginary axis 上，normal-state passive response 的耗散谱表示通常要求

\[
\operatorname{Re}\sigma_{xx}(i\xi)\ge 0,\qquad
\operatorname{Re}\sigma_{yy}(i\xi)\ge 0.
\]

因此 Stage 5.2 把显著负的对角实部标记为 `NEGATIVE_DIAGONAL`。这里采用的是宽松数值容差，而不是把非常小的浮点噪声解释为物理负电导。

## off-diagonal 与有限 q 角向结构

normal state、无显式 TRS breaking、无外磁场时，\(\sigma_{xy}\) 和 \(\sigma_{yx}\) 预期应较小，或者至少满足明确的对称关系。因此脚本报告

\[
\sqrt{|\sigma_{xy}|^2+|\sigma_{yx}|^2}
\]

相对于 diagonal norm 的比例。

不过 finite-q response 本身可能携带角向结构；后续如果进入 d-wave 或 superconducting state，也可能出现更复杂的方向依赖。因此 Stage 5.2 不把所有 large offdiag 自动判为错误，而是标记为 `OFFDIAG_LARGE_REQUIRES_SYMMETRY_AUDIT` 或 `MONITOR`，留给 Stage 5.3 做更系统的 convergence / symmetry scan。

## 为什么每个 conductivity 点仍要附带 Ward residual

conductivity 来自 physical response 的 spatial block。如果某个 response 点的 Ward residual 未闭合，那么由它得到的 conductivity 即使看起来数值平滑，也不能作为可靠电磁输入。因此 Stage 5.2 对每个 frequency、q 和积分参数组合都记录 Stage 4.18 修正约定下的 left/right Ward residual，并要求

\[
\max(\|R_{\rm left}\|,\|R_{\rm right}\|)<10^{-6}.
\]

若不满足，则该 conductivity 点标记为 `WARD_NOT_CLOSED_FOR_CONDUCTIVITY_POINT`。

## 并行执行说明

Stage 5.2 脚本支持 `--workers N`。这个参数只改变 case 调度方式，不改变 response formula、conductivity convention、Ward residual convention 或 case status 判据。

当 `--workers 1` 时，脚本按 planned case 顺序串行运行，便于调试。当 `--workers N` 且 \(N>1\) 时，脚本使用多进程并行执行各个 independent case。每个 case 仍然调用同一个 `run_case` 路径，并且仍然通过 Stage 5.1b helper

\[
\sigma^{\rm model}_{ij}(i\Omega)
=-\frac{response[1:3,1:3]_{ij}}{\Omega_{\rm eV}}
\]

生成 model-level bilayer sheet conductivity。

并行完成顺序可能与 planned order 不同，但脚本在写入 JSON/Markdown 前会按 planned case 顺序排序，因此输出稳定、可复现。建议先用 `--max-cases 1` 测量单个 case 的耗时，再用 `--workers 4` 或机器核心数的一半运行 moderate scan。

## 为什么还不能进入 reflection/Casimir

本阶段输出的是 model-level bilayer-normalized 2D sheet conductivity。它尚未应用最终 SI sheet scaling，不是 3D bulk conductivity，也不是 single-layer conductivity。它也还没有经过 reflection matrix 所需的边界条件、单位归一化、\(n=0\) policy 和 finite-thickness/slab 处理。

因此 Stage 5.2 的结论只能决定是否进入 Stage 5.3 conductivity convergence / symmetry scan。它不产生 reflection matrix，不产生 Casimir torque，也不声明 Casimir-ready。
