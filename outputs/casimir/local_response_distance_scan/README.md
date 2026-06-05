# local-response Casimir 初级结论

本目录保存基于当前 local-response contract 得到的距离扫描初级结论。它使用已通过
数值稳定性检查的推荐参数，扫描距离 `d` 下的 energy、finite-difference torque 和
zero-torque baseline。

本目录中的结果仍然满足：

```text
local_response=True
finite_momentum_resolved=False
n0_policy=skip
benchmark_only=True
preliminary_local_response_conclusion=True
not_final_casimir_conclusion=True
```

因此这里是当前可引用的 local-response 初级结论，但不是最终 Casimir torque 结论，
也不能解释为 finite-momentum Lifshitz 结果。

主要入口：

- `distance_scan_summary.md`：中文状态摘要；
- `distance_scan_command.sh`：完整 full-run 命令；
- `data/distance_scan.csv` 与 `data/distance_scan.npz`：可复现数据；
- `figures/`：距离与角度依赖的初级结论图。

当前 full distance scan 已完成，输出状态为：

```text
quick_test_only=False
full_distance_scan_completed=True
local_response=True
finite_momentum_resolved=False
n0_policy=skip
benchmark_only=True
preliminary_local_response_conclusion=True
not_final_casimir_conclusion=True
```

本次 full scan 包含 normal、spm、dwave 与 toy anisotropic plumbing control。
`energy_vs_theta_by_distance.png` 和 `torque_vs_theta_by_distance.png` 按 kind 分面，
避免多距离图例遮挡数据。运行命令记录在 `distance_scan_command.sh`，完整状态与
cache 统计记录在 `distance_scan_summary.md`。

torque 图以 `torque_tolerance=1e-20` 归一化；虚线 `|tau_fd/tau_tol|=1` 是当前
zero-torque baseline 判据。图中物理 kinds 均位于该阈值以内。
