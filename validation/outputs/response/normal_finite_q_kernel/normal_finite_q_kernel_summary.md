# Normal finite-q current-current kernel 同接口收敛检验

## 检验目的

确认 normal-state finite-q current-current kernel `K_ij(q, iOmega_n)` 在 `q=0` 与 `q!=0` 下使用同一计算接口，并检查小 q 极限的数值连续性。

## 被检验对象

- `normal_current_current_kernel_imag_axis` 对应的 finite-q current-current kernel；
- `K_xx`、`K_xy`、`K_yx`、`K_yy` 等 kernel 分量；
- C4 rotation covariance 的数值一致性。

## 检验方法与判据

- `q=0` 与 `q!=0` 均从同一 finite-q kernel 接口进入，不调用 public local sigma 分支。
- 只检查 `n>=1` 的 positive Matsubara 点。
- 不处理 `n=0` true static。
- 比较 `q=0` same-interface error、小 q same-interface error、C4 covariance error 和 kernel 分量有限性。
- 本检验不检查 Ward identity，不生成 gauge-closed finite-q conductivity，也不进入 Casimir input。

## 主要结果

- `q=0` 同接口误差为 `0`。
- 最小采样非零 q 为 `0.0001`。
- 最小非零 q 下最大 same-interface error 约为 `0.000205641`。
- 最大 C4 covariance error 约为 `6.58137e-15`。
- 所有 kernel 分量均为 finite。

## 当前判定

诊断通过：该检验可以支撑 normal-state finite-q current-current kernel 的接口一致性和基础数值稳定性。

## 对主流程的影响

- 不阻塞 local `q=0` response。
- 不证明完整 finite-q conductivity。
- 不提供 reflection input。
- 不允许作为 formal Casimir input。

## 边界说明

- `diagnostic_only`: true
- `valid_for_casimir_input`: false
- `checks_ward_validation`: false
- `checks_unit_conversion`: false
- `checks_n0_policy`: false
- `production_use_allowed`: false

## 复现入口

运行 `validation/outputs/response/normal_finite_q_kernel/command.sh`。生成的 `data/` 和 `figures/` 是 ignored artifact。

## 历史来源 / 旧 stage 对照

本检验来自早期 finite-q kernel convergence diagnostic。旧 stage 名称不再作为阅读入口。
