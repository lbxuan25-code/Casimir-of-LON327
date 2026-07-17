# Full adaptive numerical contract

## Finite outer-Q domain

径向 panel 使用父区间与二分子区间的差作为误差证据；角向使用周期全域规则的阶数倍增，并以独立 offset 做 aliasing audit。

相邻角阶数的联合误差逐 pairing、逐 Matsubara 项计算：

```text
E_radial = E_r(N_phi/2) + E_r(N_phi)
E_angular = |F(N_phi) - F(N_phi/2)|
E_joint = E_radial + E_angular
```

控制器比较径向和角向误差占各自预算的最大归一化值，只推进当前主导方向。

## Outer-Q tail

累计 cutoff 只增加外侧 shell。对 shell `k`：

```text
Delta F_k = F(u_k) - F(u_{k-1})
A_k = |Delta F_k| + E_k + E_{k-1}
```

只有连续等宽 shell 对每个 channel 都满足收缩比上限 `r_max < 1`，才使用：

```text
E_outer_tail <= A_last * r_max / (1 - r_max)
```

## Matsubara tail

cutoff `N` 总是表示连续集合 `n=0,...,N`。每项包络为：

```text
A_n = |F_n| + E_outer,n
```

连续高频窗口必须逐项满足 `A_n/A_(n-1) <= r_max`，尾界为：

```text
E_Matsubara_tail <= A_N * r_max / (1 - r_max)
```

正负项抵消不能替代逐项衰减。

## Total budget

对每个 pairing：

```text
S_N = sum(F_n)
E_finite = sum(E_outer,n)
T_total = max(atol, rtol * |S_N|)
E_total = E_finite + E_Matsubara_tail
```

有限频率、Matsubara tail 和总误差必须分别通过分配预算。

## Authorization boundary

数值合同完整不等于真实模型已获得物理授权。当前结果固定保持：

```text
production_casimir_allowed = false
```
