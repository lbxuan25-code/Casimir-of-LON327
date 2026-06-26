# Normal finite-q current-current kernel 收敛诊断

## 检验对象

本目录检验 normal-state finite-q current-current kernel `K_ij(q, iOmega_n)` 的数值一致性。它关注 kernel 层的同接口计算和有限动量行为，不等于完整 finite-q conductivity。

## 当前设置

- `q=0` 和 `q!=0` 使用同一 finite-q kernel 接口计算。
- `q=0` 分支不再调用 public local sigma。
- 默认只检查 `n>=1` 的正 Matsubara 点。
- 本阶段不处理 `n=0` true static。
- Matsubara 频率直接使用 bosonic Matsubara energy，不使用额外 `omega+eta` 展宽。
- current-current-only 不是 gauge-closed finite-q conductivity。
- Ward identity 尚未在该诊断中闭合检查。

## 关键结果

- q=0 same-interface error: `0`
- smallest sampled nonzero q: `0.0001`
- maximum same-interface error at smallest nonzero q: `0.000205641`
- maximum C4 covariance error: `6.58137e-15`
- all K components finite: `true`

## 当前结论

该诊断可作为 normal-state finite-q current-current kernel 的数值一致性证据：同一接口在 `q=0` 极限上闭合，非零小 q 的误差处于可追踪范围，C4 covariance 检查未显示明显数值破坏。

但它不代表完整 finite-q conductivity，也不代表 gauge-closed finite-q response。该结果不进入正式 Casimir input。

## 复现入口

主要命令保存在 `command.sh`。运行后生成的 `data/`、`figures/` 仍为 ignored artifact。
