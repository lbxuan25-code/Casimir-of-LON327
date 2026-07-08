# 直接有限 q TM/TE 目标基推导

本文档定义 sandbox v1 使用的全部约定。当前推导只定义模型单位中的目标基响应候选，不完成 Casimir kernel 归一化，因此不能声明 Casimir-ready。

## 原始分量基

原始源顺序固定为

```text
[A0, Ax, Ay]
```

对每个有限 q 点，

```text
q = sqrt(qx^2 + qy^2)
qhat = (qx/q, qy/q)
that = (-qhat_y, qhat_x)
```

横向方向固定为

```text
that = (-qhat_y, qhat_x)
```

不能使用相反号。若 `q < q_tol`，本路径必须报错，因为有限 q 的 TM/TE 基在 q=0 未定义。

## Gauge 系数与归一化

sandbox v1 中集中定义

```text
g0 = xi
gL = q
```

这些系数只应出现在 `tmte/theory/conventions.py`。当前归一化标签为

```text
basis_normalization = "unnormalized_gauge_orthogonal_tm_te"
```

也就是说，v1 不静默采用归一化 TM 基。

## 对齐基中的目标源

在 `(A0, AL, AT)` 对齐基中：

```text
G  = (g0,  gL, 0)
TM = (-gL, g0, 0)
TE = (0,   0,  1)
```

其中 TM 源对应纵向电场组合

```text
F_L = g0 * A_L - gL * A0
```

这是目标物理响应基，不是 Ward repair projection。

## 目标顶点

给定原始分量顶点

```text
Gamma0
Gammax
Gammay
```

先定义

```text
GammaL = qhat_x * Gammax + qhat_y * Gammay
GammaT = -qhat_y * Gammax + qhat_x * Gammay
```

再定义目标顶点

```text
GammaG  = g0 * Gamma0 + gL * GammaL
GammaTM = -gL * Gamma0 + g0 * GammaL
GammaTE = GammaT
```

行侧与列侧对同一个外部 `Q` 使用同一组 `qhat/that` 与目标基系数；有限 q 的行列方向由现有 Kubo 例程内部处理。

## 目标 contact

若现有模型提供空间 contact/diamagnetic 张量

```text
Dxx, Dxy, Dyx, Dyy
```

定义空间系数向量

```text
vG  = gL * qhat
vTM = g0 * qhat
vTE = that
```

对 `a,b in ["G", "TM", "TE"]`：

```text
C_ab = v_a^i D_ij v_b^j
```

v1 只实现空间 contact 路径，不发明标量 contact 项。

## Bare blocks 与 Schur

诊断源顺序

```text
S_diag = ["G", "TM", "TE"]
S_phys = ["TM", "TE"]
```

直接计算 bare blocks：

```text
K_SS
K_Seta
K_etaS
K_etaeta
```

其中 `S` 通常按 `["G", "TM", "TE"]` 排列。物理候选响应在 Schur 之后切出：

```text
K_TMTE_eff = K_eff[TM/TE, TM/TE]
```

Schur 公式为

```text
K_eff = K_SS - K_Seta @ inv(K_etaeta) @ K_etaS
```

实现中使用稳定 solve，不显式形成逆。

## Shifted mesh 平均

生产式顺序是先平均 bare blocks，再 Schur：

```text
Kbar_SS     = mean_s K_SS[s]
Kbar_Seta   = mean_s K_Seta[s]
Kbar_etaS   = mean_s K_etaS[s]
Kbar_etaeta = mean_s K_etaeta[s]
Kbar_eff    = Schur(Kbar_SS, Kbar_Seta, Kbar_etaeta, Kbar_etaS)
```

平均每个 shift 的 Schur 结果只允许作为可选诊断，不作为主结果。

