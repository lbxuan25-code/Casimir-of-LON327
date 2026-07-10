# Zero-Matsubara static k-grid convergence

本目录对应 exact `xi=0`、固定非零 q 的 two-band static response 收敛扫描。

当前尚未提交本地收敛结论。运行扫描后，完整 CSV、JSON 和 log 写入 `raw/`；该目录由 Git 忽略。完成扫描后，只提交人工审阅过的 `summary.md` 或小型 status 文件。

主要观测量：

- primitive/effective Ward residual；
- Schur condition number；
- imaginary residual；
- aggregate longitudinal gauge leakage；
- 分解后的 `K_0L`、`K_L0`、`K_LL`、`K_LT`、`K_TL` scaled absolute/relative entries，以及主导分量；
- `K_LL` 的 bubble、direct/contact、裸 `K_SS`、collective Schur correction 和最终 `K_eff` 分解；
- collective correction 的四个 amplitude/phase channel：`eta1-eta1`、`eta1-eta2`、`eta2-eta1`、`eta2-eta2`；
- `K_etaeta` 的奇异值、按模排序的特征值、inverse Frobenius norm，以及左右 `L-eta` coupling；
- phase bubble、Goldstone counterterm、phase total、inverse 和 factorized correction；
- bubble/direct、Schur 与 phase bubble/counterterm cancellation ratio；
- 左右 Ward RHS、collective projection、effective direct/predicted norms 与 RHS-projection cancellation ratio；
- density-transverse mixing；
- `chi_bar` 与 `Dbar_T`；
- material cache、q workspace、response 和后处理耗时；
- peak RSS。

五个 longitudinal relative entries 使用与 `StaticSheetValidation` 相同的 mixed-unit scaling 和总尺度；其二范数必须复现 `relative_longitudinal_gauge_residual`。

`K_LL` 使用固定符号约定：

```text
K_SS  = K_bubble + K_direct
K_eff = K_SS - K_collective_correction
```

扫描入口会按实际 inverse policy 独立重算
`K_collective_correction = K_Seta @ inv(K_etaeta) @ K_etaS`，并对两条重构关系执行 fail-closed 检查。

collective channel 采用固定顺序：

```text
eta1 = amplitude
eta2 = phase
```

每项定义为：

```text
term_ab = K_Leta[a] * inv(K_etaeta)[a,b] * K_etaL[b] / E0
```

四项之和必须复现 scaled local `K_LL` collective correction；不一致时扫描直接失败。

## Equal-cost quadrature comparison

`validation.run_static_quadrature_compare` 比较相同总采样点数下的两种规则：

```text
midpoint:      cell_nk = 2 * base_nk, 1 shift,  N = 4 * base_nk^2
gauss2_shift4: cell_nk = base_nk,     4 shifts, N = 4 * base_nk^2
```

`gauss2_shift4` 使用每个均匀基元胞内的二维二点 Gauss-Legendre 节点，即四个对称周期偏移
`1/2 +/- 1/(2*sqrt(3))` 的笛卡尔积。它不是 `2*base_nk` midpoint 网格的重排。

四个偏移网格的点和归一化权重会先合并，再构造一次 material/q workspace。由此 bubble、direct/contact、EM-collective mixed blocks、collective bubble 和 Goldstone counterterm 都先完成同一个 quadrature 积分，最后只执行一次 Schur complement。禁止分别计算四个 effective kernel 后再平均。

`rhs_projection_cancellation_ratio` 定义为

```text
||effective_predicted|| / max(||primitive_rhs||, ||collective_projection||, 1e-30)
```

因此该比值越小，表示 RHS 与 collective projection 的物理相消越充分；它不同于已经衡量数值恒等式闭合的 Ward residual。

## Ward mixed absolute-relative closure

生产 Ward pass/fail 使用混合判据：

```text
||residual|| <= absolute_residual_tolerance
                + residual_tolerance * reference_scale
```

`residual_tolerance` 保留为 relative tolerance；默认 `absolute_residual_tolerance=1e-12`。旧的 pure relative residual 仍保留为诊断字段，但不再单独决定 closure。

当

```text
residual_tolerance * reference_scale <= absolute_residual_tolerance
```

时，诊断标记为 denominator collapse：恒等式两边已经同时接近零，pure relative residual 可能因分母塌缩而上升。此时应检查 absolute residual 与 mixed ratio，不能把 relative residual 的上升解释为物理 Ward closure 变坏。
