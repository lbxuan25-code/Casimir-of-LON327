# finite-q BdG superconducting response 验证摘要

## 检验对象

本目录归纳 finite-q BdG superconducting response 的诊断结果。当前架构边界是：generic finite-q engine 提供有限动量响应计算骨架，`PairingAnsatz` input layer 提供 pairing-specific 结构。该分离本身是可用的，但不自动意味着 raw finite-q BdG response 已可作为正式 Casimir input。

## 已检查内容

- bare kernel：finite-q current vertex、Hermiticity、直接 kernel 数值 sanity。
- amplitude / phase collective vertices：phase-only、extended Ward、amplitude-phase restoration、commensurate-q control、多种 dwave collective vertex 尝试。
- Schur complement：混合块、解析 mixed direct、LSQ projection、k-resolved mixed block、longitudinal completion、gauge-covariant collective package。
- q=0 normal limit：`Delta -> 0` 与 normal backend 对比。
- q->0 consistency：有限 q response 向 local comparison 收敛。
- reflection input candidate：将 BdG response 转为 reflection input 的候选 gate。

## 当前 stage 结论

通过的核心诊断：

- `bare_kernel_audit`：`PASSED`
- `normal_limit`：`PASSED`
- `q0_limit`：`PASSED`

失败或仍未闭合的诊断：

- `amplitude_phase_gauge_restoration`：`FAILED`
- `reflection_input_audit`：`FAILED`
- `stageSC_2k_gauge_covariant_collective_package_audit`：`FAILED_STAGE2K_CONTROL_REGRESSION`
- 多个 stageSC_2 系列显示 partial pass 或 diagnostic-only 改善，但没有给出可升格为 production 的 Ward closure。

这些失败表示：当前 raw finite-q BdG superconducting response 的 collective-sector Ward restoration 与 reflection-input gate 尚未同时闭合。失败不阻塞 local `q=0` response 的使用边界；它阻塞 finite-q BdG response 作为正式 Casimir input。

## 明确边界

- raw finite-q BdG response 仍是 diagnostic-only。
- 当前不允许作为正式 Casimir input。
- 不使用 LSQ、response-level repair 或 quick audit 结果修正 response。
- 不因为 commensurate-q control 或某个 quick audit 通过就提升为 production result。
- 不计算或声明正式 Casimir energy、force、torque。

production 路径应读取 `bdg_finite_q_validation_status.json`。当前 marker 顶层状态为 `FAILED`，默认必须拒绝 finite-q BdG response；只有显式 diagnostic override 才能继续。

## 复现入口

主要命令保存在 `command.sh`。原始 stage 输出可由 validation scripts 重新生成，但生成物不再作为长期 Git artifact 保存。
