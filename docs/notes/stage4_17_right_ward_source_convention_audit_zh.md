# Stage 4.17 Right Ward source-side convention audit

## 1. Boundary

本阶段只做 diagnostic。它不修改主 response，不修改 bubble sign，不修改
$V_i$、$M_{ij}$、$j_i=-V_i$，不修改 source/observable split，不修改 direct contact，
不加入 fitted contact，不加入 $E^{ET}$，不进入 conductivity / reflection / Casimir。

## 2. Analytic source-side Ward identity

当前 physical response 使用

$$
J=(\rho,-V_x,-V_y),\qquad P=(\rho,V_x,V_y).
$$

left Ward contraction 使用 observable current side：

$$
R_L[\nu]=i\Omega\Pi_{0\nu}+q_x\Pi_{x\nu}+q_y\Pi_{y\nu}.
$$

但 source side 的 spatial vertex 是 $P_i=V_i$，不是 physical current $j_i=-V_i$。
finite-q band routing 给出

$$
G_+^{-1}-G_-^{-1}=i\Omega\rho-q_iV_i.
$$

因此 right Ward 最自然的 source-side contraction 是

$$
R_R^{(-)}[\mu]=i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y}.
$$

## 3. Candidate definitions

Stage 4.17 比较四个 right Ward 候选：

$$
+i\Omega\Pi_{\mu0}+q_x\Pi_{\mu x}+q_y\Pi_{\mu y},
$$

$$
+i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y},
$$

$$
-i\Omega\Pi_{\mu0}+q_x\Pi_{\mu x}+q_y\Pi_{\mu y},
$$

$$
-i\Omega\Pi_{\mu0}-q_x\Pi_{\mu x}-q_y\Pi_{\mu y}.
$$

解析预测最强候选为 `R_right_plus_omega_minus_q`。

## 4. Adaptive full-response setup

本阶段复用 Stage 4.16 的 Fermi-window adaptive quadrature 点和权重，用同一套点权重
计算完整

$$
\Pi_{\mu\nu}=\Pi_{\mu\nu}^{bubble}+D_{\mu\nu}.
$$

bubble 使用 Stage 4.13 corrected positive prefactor，direct contact 仍为
$D_{ij}=-\langle M_{ij}\rangle$。

## 5. Boundary conclusion

如果 right Ward 未闭合，不允许回退 Stage 4.13 bubble sign，也不允许改 direct
contact。下一步应聚焦 source-side Ward convention、finite-q density vertex embedding
或 source routing。
