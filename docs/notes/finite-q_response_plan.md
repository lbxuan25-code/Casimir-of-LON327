# finite-$q$ 响应闭合计划

## 0. 目的

本文档用于记录 LNO327 Casimir torque 计算中 finite-$q$ 响应问题的解决路线，避免后续开发过程中混淆 local response、finite-$q$ response、BdG response、Casimir benchmark 和最终物理结论。

当前已经完成的基础：

* local $q=0$ response baseline 已经闭合。
* BdG response contract 已修正为

$$
K_{\mathrm{total}}
= K_{\mathrm{dia}} - K_{\mathrm{para}} .
$$

* BdG paramagnetic bubble 已加入 Nambu $1/2$ prefactor。
* $\Delta_0=0$ 时，BdG normal limit 已与 normal-state kernel 数值对齐。
* local-response Casimir distance scan 已建立 normal、$s_{\pm}$、d-wave 的 zero-torque baseline。
* toy anisotropic control 能产生非零 torque，说明 Casimir 管线对各向异性响应是敏感的。

剩余问题：

当前 Casimir 积分中已经包含平面内光子动量 $\mathbf{q}_{\parallel}$，但材料响应仍采用 local 近似：

$$
\Sigma(i\xi_n,\mathbf{q}_{\parallel})
\approx
\Sigma(i\xi_n,\mathbf{0}) .
$$

finite-$q$ 任务的目标是将其替换为受控的非局域响应：

$$
\Sigma(i\xi_n,\mathbf{q}_{\parallel}) .
$$

本文档是路线图，不是计算结果。

---

## 1. finite-$q$ 的物理含义

在 Lifshitz / Casimir 几何中，电磁涨落模式由 Matsubara 频率和面内动量标记：

$$
(i\xi_n,\mathbf{q}_{\parallel}) .
$$

当前 local-response 计算实际上使用的是：

$$
R(i\xi_n,\mathbf{q}_{\parallel})
= R\!\left[
\Sigma(i\xi_n,\mathbf{0}),
\mathbf{q}_{\parallel}
\right] .
$$

也就是说，反射矩阵知道 $\mathbf{q}_{\parallel}$，但材料响应张量本身不知道
$\mathbf{q}_{\parallel}$。

finite-$q$ 问题就是要计算：

$$
R(i\xi_n,\mathbf{q}_{\parallel})
= R\!\left[
\Sigma(i\xi_n,\mathbf{q}_{\parallel}),
\mathbf{q}_{\parallel}
\right] .
$$

这很重要，因为即使 $q=0$ 响应张量是各向同性的：

$$
\Sigma_{xx}(i\xi,0)
= \Sigma_{yy}(i\xi,0),
\qquad
\Sigma_{xy}(i\xi,0)
= 0,
$$

finite-$q$ 响应仍可能包含晶格角向谐波，例如：

$$
\cos 4\phi_q,
\qquad
\sin 4\phi_q .
$$

因此，finite-$q$ 响应有可能揭示 local $q=0$ 极限中不可见的晶格方向依赖。

---

## 2. 不能跳过的验证原则

finite-$q$ response 在接入 Casimir 之前，必须先通过以下检查。

### 2.1 local limit

对 normal 和 BdG response 都必须满足：

$$
\lim_{\mathbf{q}\to\mathbf{0}}
K(i\xi,\mathbf{q})
= K(i\xi,\mathbf{0}) .
$$

这是最基础的检查。

### 2.2 BdG normal limit

在 $\Delta_0=0$ 时，BdG finite-$q$ response 必须回到 normal finite-$q$ response：

$$
K^{\mathrm{BdG}}(i\xi,\mathbf{q};\Delta_0=0)
= K^{\mathrm{normal}}(i\xi,\mathbf{q}) .
$$

这是已经通过的 local BdG-normal decomposition 的 finite-$q$ 版本。

### 2.3 $C_4$ 协变性

对于 $C_4$ 对称模型，应满足：

$$
K(i\xi,\mathcal{R}\mathbf{q})
= \mathcal{R}\,
K(i\xi,\mathbf{q})\,
\mathcal{R}^{-1} .
$$

如果这个不满足，finite-$q$ 中出现的角向结构很可能是 mesh、vertex 或坐标 convention 的数值伪效应。

### 2.4 Ward identity / 连续性方程

完整规范闭合的 finite-$q$ response 最终应包含 density-current 响应：

$$
\Pi_{\mu\nu},
\qquad
\mu,\nu=0,x,y .
$$

需要检查连续性 / Ward identity：

$$
\mathrm{i}\xi\,\Pi_{0\nu}
+
\sum_{i=x,y}q_i\Pi_{i\nu}
\approx
0 .
$$

这里写的是 $q_{\mu}=(\mathrm{i}\xi,\mathbf{q})$ 的虚频轴 convention。具体实现也可能
通过重新定义 density vertex 吸收 $\mathrm{i}$ 因子或整体负号，但必须在代码和
summary 中明确说明所采用的 convention。

如果当前阶段只计算 current-current response，则必须明确标记：

```text
current_current_only=True
ward_identity_not_yet_checked=True
```

---

## 3. 第一阶段：normal-state finite-$q$ current-current response

### 目标

先只做 normal state，不碰 BdG，不接 Casimir。

建议新增模块：

```text
src/lno327/nonlocal_response.py
```

或：

```text
src/lno327/finite_q_response.py
```

建议新增诊断脚本：

```text
validation/scripts/numerical_stability/diagnose_normal_finite_q_response.py
```

### 公式

定义 shifted momenta：

$$
\mathbf{k}_{-}
= \mathbf{k} - \frac{\mathbf{q}}{2},
\qquad
\mathbf{k}_{+}
= \mathbf{k} + \frac{\mathbf{q}}{2}.
$$

分别对角化：

$$
H_0(\mathbf{k}_{-})
|m,\mathbf{k}_{-}\rangle
= E_m^{-}
|m,\mathbf{k}_{-}\rangle ,
$$

$$
H_0(\mathbf{k}_{+})
|n,\mathbf{k}_{+}\rangle
= E_n^{+}
|n,\mathbf{k}_{+}\rangle .
$$

定义：

$$
\Delta E_{mn}
= E_m^{-} - E_n^{+},
$$

$$
\Delta f_{mn}
= f(E_m^{-}) - f(E_n^{+}) .
$$

第一版诊断可以使用 midpoint velocity approximation：

$$
J^{\alpha}_{mn}(\mathbf{k},\mathbf{q})
= \left\langle
m,\mathbf{k}_{-}
\middle|
v_{\alpha}(\mathbf{k})
\middle|
n,\mathbf{k}_{+}
\right\rangle ,
$$

其中：

$$
v_{\alpha}(\mathbf{k})
= \frac{\partial H_0(\mathbf{k})}{\partial k_{\alpha}} .
$$

然后 finite-$q$ positive current-current bubble 写为：

$$
K_{\alpha\beta}^{\mathrm{para}}(i\xi,\mathbf{q})
= \sum_{\mathbf{k},m,n}
w_{\mathbf{k}}
F_{mn}(i\xi,\mathbf{q})
J^{\alpha}_{mn}(\mathbf{k},\mathbf{q})
J^{\beta}_{nm}(\mathbf{k},-\mathbf{q}) .
$$

其中：

$$
F_{mn}(i\xi,\mathbf{q})
= -
\frac{
\Delta f_{mn}\Delta E_{mn}
}{
(\Delta E_{mn})^2+\xi^2
} .
$$

当 $\mathbf{q}\to\mathbf{0}$ 且 $m=n$ 时，上式必须取解析 intraband 极限，
不能把 $\Delta f_{mm}=\Delta E_{mm}=0$ 直接作为零贡献跳过。

### 必须记录的 metadata

所有输出必须包含：

```text
finite_momentum_resolved=True
normal_state=True
current_current_only=True
midpoint_vertex_approximation=True
not_peierls_exact_vertex=True
ward_identity_not_yet_checked=True
not_final_casimir_input=True
```

### 必须输出的诊断量

至少输出：

```text
omega_eV
q_model
q_angle
nk
temperature_K
eta_eV
K_xx, K_yy, K_xy, K_yx
q_to_zero_relative_error
C4_covariance_error
angular_harmonic_cos4
angular_harmonic_sin4
```

### 验收标准

* $q\to0$ 极限回到已有 local normal response。
* $C_4$ covariance error 足够小。
* 随 $n_k$ 增大结果稳定。
* 该阶段不运行任何 Casimir 计算。

---

## 4. 第二阶段：Casimir $q$-grid 到 model-$q$ 的单位审计

### 目标

搞清楚 Casimir 积分中实际采样到了哪些 finite-$q$。

Casimir 积分中常用：

$$
u = q_{\parallel}d .
$$

因此：

$$
q_{\mathrm{physical}}
= \frac{u}{d}.
$$

电子模型 Hamiltonian 使用无量纲晶格动量，所以：

$$
q_{\mathrm{model}}
= a q_{\mathrm{physical}}
= a\frac{u}{d},
$$

其中 $a$ 是面内晶格常数。

建议新增脚本：

```text
validation/scripts/numerical_stability/audit_casimir_q_to_model_q.py
```

### 必须输出

对每个距离和积分 cutoff 输出：

```text
distance_m
u_min
u_max
du
q_physical_min_m_inv
q_physical_max_m_inv
lattice_constant_m
q_model_min
q_model_max
q_model_typical
```

### 验收标准

* 代码中显式记录面内晶格常数 $a$。
* 明确写出转换关系：

$$
q_{\mathrm{model}}
= a\frac{u}{d}.
$$

* 在 finite-$q$ response 接入 Casimir 前，必须知道最大采样 $q_{\mathrm{model}}$。
* 不允许隐藏 SI 单位与 model unit 的转换。

---

## 5. 第三阶段：BdG finite-$q$ bare bubble 诊断

### 目标

在 normal finite-$q$ response 通过后，再实现 BdG finite-$q$ bare bubble。该阶段只做诊断，不接 Casimir。

建议新增脚本：

```text
validation/scripts/numerical_stability/diagnose_bdg_finite_q_normal_limit.py
```

### 公式

定义 BdG shifted momenta：

$$
\mathbf{k}_{-}
= \mathbf{k} - \frac{\mathbf{q}}{2},
\qquad
\mathbf{k}_{+}
= \mathbf{k} + \frac{\mathbf{q}}{2}.
$$

对角化：

$$
\mathcal{H}_{\mathrm{BdG}}(\mathbf{k}_{-})
|a,\mathbf{k}_{-}\rangle
= E_a^{-}
|a,\mathbf{k}_{-}\rangle ,
$$

$$
\mathcal{H}_{\mathrm{BdG}}(\mathbf{k}_{+})
|b,\mathbf{k}_{+}\rangle
= E_b^{+}
|b,\mathbf{k}_{+}\rangle .
$$

使用与当前 local response 一致的 BdG charge-current convention：

$$
J_{\alpha}^{\mathrm{BdG}}
= \begin{pmatrix}
\partial_{\alpha}H_0(\mathbf{k}) & 0 \\
0 & -\partial_{\alpha}H_0^{T}(-\mathbf{k})
\end{pmatrix}.
$$

第一版诊断允许使用 midpoint vertex approximation：

$$
J^{\alpha}_{ab}(\mathbf{k},\mathbf{q})
= \left\langle
a,\mathbf{k}_{-}
\middle|
J_{\alpha}^{\mathrm{BdG}}(\mathbf{k})
\middle|
b,\mathbf{k}_{+}
\right\rangle .
$$

然后：

$$
K_{\alpha\beta}^{\mathrm{BdG,para}}(i\xi,\mathbf{q})
= \frac{1}{2}
\sum_{\mathbf{k},a,b}
w_{\mathbf{k}}
F_{ab}^{\mathrm{BdG}}(i\xi,\mathbf{q})
J^{\alpha}_{ab}(\mathbf{k},\mathbf{q})
J^{\beta}_{ba}(\mathbf{k},-\mathbf{q}) .
$$

其中

$$
\Delta E_{ab}^{\mathrm{BdG}}
= E_a^{-}-E_b^{+},
\qquad
\Delta f_{ab}^{\mathrm{BdG}}
= f(E_a^{-})-f(E_b^{+}),
$$

以及

$$
F_{ab}^{\mathrm{BdG}}(i\xi,\mathbf{q})
= -
\frac{
\Delta f_{ab}^{\mathrm{BdG}}
\Delta E_{ab}^{\mathrm{BdG}}
}{
\left(\Delta E_{ab}^{\mathrm{BdG}}\right)^2+\xi^2
}.
$$

与 normal-state bubble 相同，$\mathbf{q}\to\mathbf{0}$ 下的同带项必须使用
intraband 极限。

其中 $1/2$ 是必须保留的 Nambu redundancy prefactor。local response 已经证明，不加这个因子会导致 BdG paramagnetic bubble 在 $\Delta_0=0$ 时比 normal 结果多一倍。

### 必须记录的 metadata

```text
bdg_finite_q=True
bare_bubble=True
current_current_only=True
midpoint_vertex_approximation=True
collective_phase_correction=False
ward_identity_not_yet_checked=True
not_final_casimir_input=True
```

### 必须输出的诊断量

* $q\to0$ local-limit mismatch。
* $\Delta_0=0$ normal-limit mismatch。
* $C_4$ covariance error。
* spm/dwave 在相同 $q$ grid 下的对比。
* angular harmonic extraction。

### 验收标准

* $\Delta_0=0$ 时 BdG finite-$q$ response 匹配 normal finite-$q$ response。
* $q\to0$ 时 BdG finite-$q$ response 匹配 local BdG response。
* $C_4$ covariance 通过。
* 该阶段不做 Casimir 结论。

---

## 6. 第四阶段：density-current response 与 Ward identity

### 目标

将 response 从 current-current $K_{ij}$ 扩展为规范更完整的 density-current tensor：

$$
\Pi_{\mu\nu},
\qquad
\mu,\nu=0,x,y .
$$

### 至少包含的分量

```text
Pi_00
Pi_0x, Pi_0y
Pi_x0, Pi_y0
Pi_xx, Pi_xy, Pi_yx, Pi_yy
```

### 要检查的恒等式

检查 finite-$q$ 连续性约束：

$$
\mathrm{i}\xi\,\Pi_{0\nu}
+
q_x\Pi_{x\nu}
+
q_y\Pi_{y\nu}
\approx
0 .
$$

具体符号 convention 必须写入文档和 summary；如果 density vertex 的定义吸收了
$\mathrm{i}$ 因子，则应同步写出等价的实数形式。

### 可能遇到的问题

对于 superconducting BdG response，bare bubble 可能不满足 Ward identity。这时可能需要 collective phase / vertex correction。

如果出现这种情况，输出必须明确写：

```text
bare_bubble_ward_identity_failed=True
collective_correction_required=True
```

### 验收标准

* normal-state finite-$q$ response 在数值容差内满足 Ward identity。
* BdG finite-$q$ response 要么满足 Ward identity，要么明确定位缺失的 collective correction。
* 在这一阶段没搞清楚前，不把 finite-$q$ Casimir benchmark 当成最终结论。

---

## 7. 第五阶段：finite-$q$ response cache 与 Casimir 积分接入

### 目标

只有在前面各阶段通过后，才将 finite-$q$ response 接入 Casimir 积分。

建议新增 benchmark：

```text
validation/scripts/casimir/benchmark_casimir_finite_q_response_distance_scan.py
```

### response cache key 必须扩展

finite-$q$ cache key 必须包含：

```text
kind
omega_eV
qx_model
qy_model
temperature_K
eta_eV
nk
delta0
vertex_convention
finite_q_mode
ward_identity_status
```

不能复用 local-response cache。

### Casimir 积分中的 $q$ 映射

对每个积分点：

$$
q_x^{\mathrm{physical}}
= \frac{u}{d}\cos\phi ,
$$

$$
q_y^{\mathrm{physical}}
= \frac{u}{d}\sin\phi .
$$

然后：

$$
q_x^{\mathrm{model}}
= a q_x^{\mathrm{physical}},
$$

$$
q_y^{\mathrm{model}}
= a q_y^{\mathrm{physical}} .
$$

### 必须记录的 metadata

```text
finite_momentum_resolved=True
local_response=False
n0_policy=skip
benchmark_only=True
not_final_casimir_conclusion=True
```

### 必须对比的 benchmark

* local-response baseline vs finite-$q$ response。
* normal / spm / dwave zero-torque baseline。
* toy anisotropic control。
* $C_4$-covariant finite-$q$ angular harmonics。
* response cache correctness。

### 验收标准

* finite-$q$ benchmark 不使用 local cache。
* $q\to0$ limit 与 local benchmark 对齐。
* 对称性保持的情形不产生伪 torque。
* toy anisotropic control 仍能产生 torque。
* summary 清楚写明剩余限制。

---

## 8. 第六阶段：各向异性机制 benchmark

### 目标

在 finite-$q$ response 技术和物理上都通过验证后，引入可控的 $C_4$-breaking 机制。

可能的机制包括：

```text
orbital anisotropy
strain-like hopping anisotropy
external-field-induced C4 breaking
nematic perturbation
pairing-sector anisotropy
surface-orientation anisotropy
```

### 必须测试

对每一种机制，至少做：

```text
anisotropy_strength_scan
spm_vs_dwave_comparison
torque_vs_distance
torque_vs_angle
response_tensor_anisotropy
C4-breaking source documented
```

### 必须区分的物理含义

输出必须区分：

```text
toy anisotropy
externally induced anisotropy
intrinsic spontaneous anisotropy
finite-q crystal harmonic anisotropy
```

只有后三类可以作为物理机制讨论。

---

## 9. 禁止事项

不要：

* 在 normal finite-$q$ 验证前直接把 finite-$q$ 接入 Casimir；
* 复用 local-response cache；
* 在 $q\to0$、normal limit、$C_4$ covariance 通过前声称 finite-$q$ Casimir 结论；
* 把 current-current-only response 当成完整规范闭合 response；
* 混淆 midpoint vertex approximation 和 Peierls-exact vertex；
* 隐藏 model $q$ 与 SI $q$ 的转换；
* 认为 finite-$q$ 自动解决 $n=0$ Matsubara policy。

---

## 10. 当前推荐的下一步任务

从第一阶段开始：

```text
实现 normal-state finite-q current-current response diagnostic。
```

建议文件：

```text
src/lno327/nonlocal_response.py
validation/scripts/numerical_stability/diagnose_normal_finite_q_response.py
```

第一验收目标：

```text
q -> 0 normal finite-q response matches local normal response.
C4 covariance passes.
No Casimir code modified.
```

只有这一步通过后，才开始 BdG finite-$q$。
