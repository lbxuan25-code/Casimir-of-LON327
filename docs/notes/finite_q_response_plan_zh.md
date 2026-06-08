# finite-q 响应路线图

本文是 finite-q / nonlocal response 开发的 single source of truth。当前目标不是直接给出
finite-q conductivity 或 Casimir 材料结论，而是把对象、单位、Ward identity 和 benchmark
逐步拆开，避免把不完整的 current-current block 当成 gauge-closed response。

## 当前状态

Stage 1 已有可运行的 normal-state diagnostic：

```text
src/lno327/nonlocal_response.py
validation/scripts/numerical_stability/diagnose_normal_finite_q_response.py
validation/outputs/response/normal_finite_q_kernel_convergence/
```

当前正式输出只检查 normal finite-q current-current kernel $K(i\omega_n,\mathbf{q})$
在 $\mathbf{q}\to0$ 时是否收敛到同一接口下的 $K(i\omega_n,\mathbf{0})$。
旧 mixed sigma/K diagnostics 已删除，不再作为 validation。

## 为什么必须分阶段

有限动量响应的完整对象不是单独的 current-current kernel，而是 density-current response
张量 $\Pi_{\mu\nu}(i\omega_n,\mathbf{q})$。$K_{jj}$ 只是 current-current block。
在 local 或横向极限中，$K_{jj}$ 与 conductivity 可以有部分关系；但不能把
current-current block 除以 frequency 当成通用 finite-q 完整 conductivity。
进入 reflection/Casimir 前必须先处理单位映射、Ward identity 和 benchmark。

## 阶段路线图

### 阶段 1：正常态 finite-q current-current kernel 收敛

对象：

$$
K(i\omega_n,\mathbf{q})
$$

默认只使用 positive Matsubara $n\ge1$：

$$
\omega_n = 2\pi n k_B T .
$$

目标是检查：

$$
K(i\omega_n,\mathbf{q}) \to K(i\omega_n,\mathbf{0})
\quad \text{as} \quad \mathbf{q}\to\mathbf{0}.
$$

这里不是检查 $K(\mathbf{q})\to\sigma(\mathbf{0})$。$\mathbf{q}=\mathbf{0}$ 与
$\mathbf{q}\ne\mathbf{0}$ 都必须通过同一个 K 接口返回 current-current kernel。
public local sigma 不参与 Stage 1 正式输出；如需比较，应另开独立 debug 脚本，
不进入正式 validation outputs，也不参与 pass/fail。$n=0$ true static 不在
Stage 1 处理。

当前主判据：

$$
\epsilon_q =
\frac{\left\|K(i\omega_n,\mathbf{q}) - K(i\omega_n,\mathbf{0})\right\|}
{\left\|K(i\omega_n,\mathbf{0})\right\|}.
$$

C4 检查：

$$
K(R\mathbf{q}) \quad \text{vs.} \quad R K(\mathbf{q}) R^T .
$$

### 阶段 2：Casimir q-grid 到 model-q 单位审计

把 Casimir 积分中的无量纲变量 $u$ 与层间距离 $d$ 映射到模型动量：

$$
q_{\mathrm{model}} = a_{\parallel} \frac{u}{d}.
$$

这里 $a_{\parallel}$ 是面内赝四方 / Ni-Ni 有效晶格常数，用作从 SI 动量到 model-q
单位的面内换算长度；不要把它写成笼统 conventional crystallographic lattice constant。
$d$ 必须使用同一长度单位。Stage 2 只做单位和采样范围审计，不计算 response，
不产生 finite-q conductivity，也不声明 finite-q Casimir 结论。

Stage 2.5 在同一 audit 中加入 $a_{\parallel}$ sensitivity，例如
$a_{\parallel}=3.75, 3.85, 3.90, 3.95$ Angstrom，检查 Casimir q-grid 可访问的
$q_{\mathrm{model,max}}$ 对面内换算长度的敏感性。该敏感性仍然只是单位/采样审计，
不接入 response tensor、reflection matrix 或 Casimir integral。

### 阶段 3：BdG finite-q current-current kernel

实现 BdG finite-q current-current kernel，并检查两个极限：

$$
\Delta_0 \to 0
\quad \Rightarrow \quad
\text{normal finite-q kernel limit},
$$

以及

$$
\mathbf{q}\to\mathbf{0}
\quad \Rightarrow \quad
\text{local BdG kernel limit}.
$$

Stage 3 仍然只是 $K_{jj}$ block，不是 gauge-closed finite-q conductivity。

### 阶段 4：density-current response 与 Ward identity

构造或审计完整的

$$
\Pi_{\mu\nu}(i\omega_n,\mathbf{q})
$$

并检查连续性方程 / Ward identity。current-current-only 不能直接作为 gauge-closed finite-q
conductivity；如果 BdG bare bubble 不满足 Ward identity，需要明确缺失的 collective
phase 或 vertex correction。

### 阶段 5：reflection/Casimir benchmark 接入

只有在 Stage 4 边界清楚后，才把 finite-q response 接入 reflection/Casimir benchmark。
response cache 必须至少包含：

```text
qx_model
qy_model
matsubara_n
frequency_mode
temperature_K
response_kind
unit_convention
```

这一阶段仍是 benchmark，不直接声明最终材料结论。

### 阶段 6：各向异性机制 benchmark

比较 finite-q crystal harmonic、pairing symmetry、normal-state anisotropy 和 superconducting
kernel correction 对 torque-like observables 的贡献。只有在前面阶段通过后，才讨论材料机制。

## 当前 Stage 1 的准确任务

当前脚本只回答一个问题：

normal finite-q current-current kernel $K(i\omega_n,\mathbf{q})$ 是否在
$\mathbf{q}\to\mathbf{0}$ 时收敛到同一接口下的
$K(i\omega_n,\mathbf{0})$。

推荐 quick smoke：

```bash
python validation/scripts/numerical_stability/diagnose_normal_finite_q_response.py --quick
```

正式输出目录：

```text
validation/outputs/response/normal_finite_q_kernel_convergence/
```

## 禁止事项

- 不要用 public local sigma 填充 $\mathbf{q}=\mathbf{0}$ 的 K 字段。
- 不要把 $\omega_{\mathrm{eV}}=0$ 加数值阈值当成 true static。
- 不要在 Stage 1 默认混入 $n=0$。
- 不要把 current-current block 除以 frequency 写成 finite-q 完整 conductivity。
- 不要把 current-current-only diagnostic 接入 reflection/Casimir 作为最终输入。
- 不要从 Stage 1 输出 finite-q conductivity 或 Casimir 结论。

## 下一步

Stage 1 当前只需要继续做 quick/smoke 级别检查和代码一致性维护。下一项实质开发应进入
Stage 2：审计 Casimir q-grid 与 model-q 的单位映射
$q_{\mathrm{model}} = a_{\parallel} u / d$，并给出可复现的 sampling range 和
$a_{\parallel}$ sensitivity 报告。
