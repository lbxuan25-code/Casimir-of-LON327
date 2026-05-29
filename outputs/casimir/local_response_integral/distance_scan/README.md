# Local-response distance scan 输出目录

本目录用于保存 local-response distance scan benchmark 的输出。该 benchmark 使用已通过
数值稳定性检查的推荐参数，扫描距离 `d` 下的 local-response energy、finite-difference
torque 和 zero-torque baseline。

本目录中的结果仍然满足：

```text
local_response=True
finite_q_resolved=False
n0_policy=skip
benchmark_only=True
not_final_casimir_conclusion=True
```

因此这里不是正式 Casimir torque 结论，也不能解释为 finite-q Lifshitz 结果。

主要入口：

- `distance_scan_summary.md`：中文状态摘要；
- `distance_scan_command.sh`：完整 full-run 命令；
- `data/distance_scan.csv` 与 `data/distance_scan.npz`：可复现数据；
- `figures/`：距离与角度依赖的诊断图。

完整 distance scan 应由本地终端运行：

```bash
python scripts/benchmark_casimir_local_response_distance_scan.py --dry-run
```

先检查命令，再运行 `distance_scan_command.sh`。
