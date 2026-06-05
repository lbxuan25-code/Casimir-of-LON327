# 数值稳定性阶段输出入口

本目录是数值稳定性阶段的阅读入口说明，不存放新的大型结果文件。

权威摘要文件位置：

- `docs/notes/numerical_stability_summary.md`
- `outputs/casimir/local_response_distance_scan/distance_scan_summary.md`
- `validation/outputs/response/normal_finite_q_kernel_convergence/normal_finite_q_kernel_convergence_summary.md`

推荐阅读顺序：

1. `docs/notes/numerical_stability_summary.md`
2. `outputs/casimir/local_response_distance_scan/distance_scan_summary.md`
3. `validation/outputs/response/normal_finite_q_kernel_convergence/normal_finite_q_kernel_convergence_summary.md`

大型 `.npz`、`.csv`、`.png` 文件是可复现数据和绘图材料，不是主要阅读入口。

本阶段不使用 `.gitignore` 隐藏结果。旧 mixed sigma/K diagnostics 已删除，
不作为 validation evidence。

后续 local-response distance scan 应另开目录：

```text
outputs/casimir/local_response_distance_scan/
```

该后续阶段仍应明确标注：local-response benchmark 不是正式 finite momentum Casimir 结论。
