# finite-q BdG response validation

本目录保存 finite-q BdG superconducting response 的轻量验证证据。

## 本目录验证什么

- 裸有限 q BdG kernel 是否可计算、有限并满足基础数值一致性；
- 振幅/相位集体模 Schur restoration 是否能闭合相关 Ward residual；
- `Delta -> 0` normal limit 是否回到 normal backend；
- `q -> 0` local limit 是否连续；
- reflection input candidate 是否可作为正式下游输入。

## 本目录不验证什么

- 不计算正式 Casimir energy、force 或 torque；
- 不使用 LSQ 或 response-level repair 作为 production 方法；
- 不把 quick audit 或 commensurate-q control 通过提升为 production-ready；
- 不绕过 upstream Ward / unit / `n=0` policy gate。

## production relevance

本目录是 finite-q BdG response 的 production gate。当前 `bdg_finite_q_validation_status.json` 顶层状态为 `FAILED`，因此 raw finite-q BdG response 不能作为 formal Casimir input。

## diagnostic-only

当前 raw finite-q BdG response 是 diagnostic-only。local `q=0` response 不由本目录阻塞。

## 文件

- `bdg_finite_q_validation_summary.md`：按具体检验对象写成的中文结论。
- `bdg_finite_q_validation_status.json`：机器可读 gate marker。
- `command.sh`：复现入口。

旧 `stageSC_*` 输出已合并到 summary/status；stage 名称只在历史来源对照中保留。
