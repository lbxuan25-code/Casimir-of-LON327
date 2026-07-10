# Zero-Matsubara static k-grid convergence

本目录对应 exact `xi=0`、固定非零 q 的 two-band static response 收敛扫描。

当前尚未提交本地收敛结论。运行 `command.sh` 后，完整 CSV、JSON 和 log 写入 `raw/`；该目录由 Git 忽略。完成扫描后，只提交人工审阅过的 `summary.md` 或小型 status 文件。

主要观测量：

- primitive/effective Ward residual；
- Schur condition number；
- imaginary residual；
- aggregate longitudinal gauge leakage；
- 分解后的 `K_0L`、`K_L0`、`K_LL`、`K_LT`、`K_TL` scaled absolute/relative entries，以及主导分量；
- `K_LL` 的 bubble、direct/contact、裸 `K_SS`、collective Schur correction 和最终 `K_eff` 分解；
- bubble/direct 与 Schur 两级 cancellation ratio；
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

`rhs_projection_cancellation_ratio` 定义为

```text
||effective_predicted|| / max(||primitive_rhs||, ||collective_projection||, 1e-30)
```

因此该比值越小，表示 RHS 与 collective projection 的物理相消越充分；它不同于已经衡量数值恒等式闭合的 Ward residual。
