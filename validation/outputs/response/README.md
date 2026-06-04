# response 输出入口

本目录保存 response contract、static policy、单位转换和 numerical stability
相关验证结果。当前新增的 finite-q 内容仅限第一阶段 normal-state
current-current diagnostic，不是完整 gauge-closed response，也不是 Casimir 输入。

当前 local-response / Casimir benchmark 的主要阅读入口：

- `validation/outputs/response/bdg_static_gauge_closure/`
- `validation/outputs/response/normal_finite_q_response/`
- `outputs/casimir/local_response_distance_scan/`
- `validation/outputs/archive/response/local_sheet_imag/`
- `validation/outputs/archive/response/static_policy_comparison/`
- `validation/outputs/archive/response/unit_audit/`

`bdg_static_gauge_closure/` 检查 local BdG `K_dia - K_para` 的静态 stiffness、
候选 `rho_s` 的有限性和 C4/offdiag 行为；它不是最终 optical conductivity，
也不是最终 Casimir 输入。

`normal_finite_q_response/` 使用 q=0 local fallback 和 q!=0 midpoint-vertex bare
bubble 检查 C4 covariance、角向 harmonic 与 q-to-zero 行为。Ward identity 尚未
检查，不能将其描述为最终 finite-q conductivity 或 finite-q Casimir input。
