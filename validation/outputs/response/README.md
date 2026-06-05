# response 输出入口

本目录保存 response contract、static policy、单位转换和 numerical stability
相关验证结果。当前新增的 finite-q 内容仅限第一阶段 normal-state
current-current diagnostic，不是完整 gauge-closed response，也不是 Casimir 输入。

当前 local-response / Casimir benchmark 的主要阅读入口：

- `validation/outputs/response/bdg_static_gauge_closure/`
- `validation/outputs/response/normal_finite_q_kernel_convergence/`
- `outputs/casimir/local_response_distance_scan/`

`bdg_static_gauge_closure/` 检查 local BdG `K_dia - K_para` 的静态 stiffness、
候选 `rho_s` 的有限性和 C4/offdiag 行为；它不是最终 optical conductivity，
也不是最终 Casimir 输入。

`normal_finite_q_kernel_convergence/` 是当前正式的第一阶段 normal finite-q
current-current kernel diagnostic。q=0 与 q!=0 都由同一 K 接口计算，用于检查
K(q)->K(0) 的 same-interface 收敛和 C4 covariance。它只覆盖 n>=1 positive
Matsubara；n=0 true static、Ward identity、gauge-closed finite-q conductivity
和 Casimir 接入均不在本阶段处理。
finite-q 后续路线以 `docs/notes/finite_q_response_plan_zh.md` 为准。
