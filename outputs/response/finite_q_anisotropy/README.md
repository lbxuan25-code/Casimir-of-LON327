# finite-q response 角向各向异性诊断输出

本目录保存 finite-q response anisotropy diagnostic / prototype 的输出。该阶段只检查
response 层的

```text
sigma_ab(i xi_n, q, phi_q)
```

是否在有限 dimensionless BZ momentum `q_magnitude` 下出现角向结构，以及 spm/dwave
相对 normal 的差异是否有可见 contrast。

这里的 `q_magnitude` 与 k 网格一样使用无量纲 BZ momentum，不是 SI wavevector。

本目录不是 Casimir 结果目录。当前限制为：

- `gauge_status=prototype_not_ward_verified`
- finite-q diamagnetic/Ward identity 尚未严格闭合
- n=0 zero-frequency model 仍未完成
- `final_casimir_input=False`
- `not_final_Casimir_conclusion=True`

主要文件：

- `data/finite_q_anisotropy.csv`
- `data/finite_q_anisotropy.npz`
- `finite_q_anisotropy_summary.md`
- `figures/response_xx_vs_phi.png`
- `figures/A4_vs_q.png`
- `figures/pairing_contrast_vs_q.png`
- `figures/local_limit_error.png`
