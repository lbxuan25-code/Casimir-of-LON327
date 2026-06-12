# Stage 5.1 response-to-conductivity convention audit

## 目的

Stage 4 已经验证 normal-state density/current response 的 Ward identity 诊断约定，但这并不等于已经得到可用于 reflection 或 Casimir 的 conductivity。Stage 5.1 单独审计

\[
\Pi_{ij}(i\xi)\rightarrow\sigma_{ij}(i\xi)
\]

的符号、单位和代码路径。

## 为什么不能直接进入 Casimir

Ward closure 说明当前 response kernel 在指定 source/observable convention 下满足规范一致性诊断。Casimir / reflection 层需要的是电磁边界条件中的 conductivity，通常还要明确 sheet conductivity、dimensionless reflection conductivity、Matsubara frequency convention、\(n=0\) policy 和单位归一化。

因此 response 通过 Ward identity 后，还必须单独确认 \(\Pi_{ij}\) 与 \(\sigma_{ij}\) 的关系。

## response convention

当前 physical response 使用

\[
\Pi_{\mu\nu}=\frac{\delta\langle J_\mu\rangle}{\delta a_\nu},
\qquad
a_\nu=(\phi,A_x,A_y),
\]

\[
J=(\rho,j_x,j_y)=(\rho,-V_x,-V_y),
\qquad
P=(\rho,V_x,V_y).
\]

因此 spatial block 应解释为

\[
\Pi_{ij}=\frac{\delta\langle j_i\rangle}{\delta A_j},
\]

而不是 \(V_iV_j\) 的 source-source block。

## conductivity convention ambiguity

从

\[
j_i(i\xi)=\Pi_{ij}(i\xi)A_j(i\xi)
\]

到 conductivity 需要知道 Matsubara / Euclidean convention 下 \(E_j\) 与 \(A_j\) 的关系。例如：

\[
E_j(i\xi)=+\xi A_j(i\xi),
\qquad
\sigma_{ij}(i\xi)=\frac{\Pi_{ij}(i\xi)}{\xi},
\]

或

\[
E_j(i\xi)=-\xi A_j(i\xi),
\qquad
\sigma_{ij}(i\xi)=-\frac{\Pi_{ij}(i\xi)}{\xi},
\]

也可能写作

\[
E_j(i\Omega)=i\Omega A_j(i\Omega),
\qquad
\sigma_{ij}(i\Omega)=\frac{\Pi_{ij}(i\Omega)}{i\Omega}.
\]

当前仓库已有 local Kubo conductivity 和 sheet normalization helper，但 finite-q physical \(\Pi_{ij}\rightarrow\sigma_{ij}\) 的 Euclidean \(E/A\) 符号不能只从代码唯一确定。因此 Stage 5.1 将其标记为 `CONVENTION_NOT_UNIQUELY_DETERMINED_FROM_CODE`，等待显式约定。

## 2D sheet 与 3D bulk

`response_units.py` 当前采用 2D sheet conductivity convention：

\[
\sigma_{\mathrm{sheet,SI}}=(e^2/\hbar)\sigma_{\mathrm{model}}.
\]

这不同于 3D bulk conductivity。若后续 reflection 层需要 sheet response，则应继续使用 sheet conductivity；若需要 bulk response，则必须额外定义层厚、层间距或体积归一化。

## 后续

Stage 5.2 应先做 numerical conductivity sanity check，确认选定的 \(\Pi\rightarrow\sigma\) convention、单位归一化和 \(n=0\) policy。只有 conductivity 层也通过后，才可以考虑 reflection/Casimir 接口验证。

