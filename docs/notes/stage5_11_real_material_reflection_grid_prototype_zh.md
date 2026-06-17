# Stage 5.11 real-material reflection-grid prototype

## 为什么从 toy integration 后进入小批真实材料点

Stage 5.10 已经证明 toy 层的 Matsubara 求和和 \((Q,\varphi)\) 积分器可以运行。下一步需要确认真实 LNO327 response 链条能在一小批离散 \((n,Q,\varphi)\) 点上接入同一套 reflection 和 trace-log integrand convention。

Stage 5.11 因此只做 small real-material discrete-point prototype，不做 production grid。

## 与正式 Casimir energy 的区别

本阶段只计算每个离散点的链条：

\[
\Pi_{\mu\nu}
\rightarrow
\sigma^{model}_{xy}
\rightarrow
\tilde\sigma_{xy}
\rightarrow
\tilde\sigma_{LT}
\rightarrow
R_E^{LT}
\rightarrow
R^{TE/TM}
\rightarrow
\log\det M.
\]

这里的 \(\log\det M\) 是 pointwise identical-sheet hook-in check，不是 Matsubara sum，也不是 \(Q\)-integral。因此它不是真实 Casimir energy、force 或 torque。

## 为什么排除 \(n=0\) 和 \(Q=0\)

Stage 5.11 默认只使用 \(n>0\)，因为 \(n=0\) zero-mode 需要单独审计。

同时默认只使用 \(Q>0\)，因为 \(Q=0\) 时面内方向未定义，\(L/T\) 与 TE/TM 的面内方向也不唯一。正式生产计算需要单独处理这个极限或用对称性处理。

## 从 \((n,Q,\varphi)\) 到 \((q_x,q_y)\)

输入点使用 SI 面内波矢：

\[
Q_x=Q\cos\varphi,\qquad Q_y=Q\sin\varphi.
\]

其中

\[
1\ {\rm nm}^{-1}=10^9\ {\rm m}^{-1}.
\]

模型动量由薄膜晶格常数给出：

\[
q_x^{model}=Q_x a_x,\qquad q_y^{model}=Q_y a_y.
\]

当前使用 LNO327 thin-film in-plane lattice convention，\(a_x=a_y=3.754\times10^{-10}\ {\rm m}\)。

## 从 response 到 reflection

response 使用既有 physical-current finite-q pipeline，不修改 response formula、bubble sign、direct contact 或 source/observable split。

conductivity 使用 Stage 5.1b convention：

\[
\sigma^{model}_{ij}=-\Pi_{ij}/\Omega_{\rm eV}.
\]

dimensionless sheet conductivity 使用 Stage 5.4b 单位链。reflection input 和 TE/TM adapter 分别复用 Stage 5.5b 和 Stage 5.6 helper。

## 为什么记录 Ward residual

每个真实材料点都记录 corrected Ward residual：

\[
R_L[\nu]=i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu},
\]

\[
R_R[\mu]=i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y}.
\]

这用于判断该离散 response 点是否足够稳定，可以继续进入后续材料网格规划。

## identical-sheet logdet 只是 hook-in check

本阶段使用

\[
M=I-e^{-2\kappa d}R^{TE/TM}R^{TE/TM}
\]

计算 identical-sheet pointwise \(\log\det M\)。这只说明真实材料 reflection matrix 能接入 trace-log integrand helper，不是物理能量。

## point-level 并行运行

脚本支持 `--workers N` 对不同 \((n,Q,\varphi)\) 点做多进程并行。这个并行是 grid point 级别的 multiprocessing，不是 BLAS/OpenMP threading。默认 `--workers 1` 保持顺序运行；例如 `--workers 8` 会尝试用 8 个进程并行计算不同离散点，并在输出 JSON/MD 中保持原始点顺序。

为了避免“多进程乘以多线程”的过度占用，建议并行点计算前设置：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

然后再运行：

```bash
python validation/scripts/response/stage5_11_real_material_reflection_grid_prototype.py \
  --smoke \
  --workers 8
```

## 下一步

后续仍需 material grid convergence strategy，包括更多 \((n,Q,\varphi)\) 覆盖、response-cost audit、插值策略、\(Q=0\) 与 \(n=0\) 单独审计，以及真实材料 production grid 的收敛验证。
