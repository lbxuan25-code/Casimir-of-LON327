# 数值稳定性阶段输出入口

本目录是数值稳定性阶段的阅读入口说明，不存放新的大型结果文件。

权威摘要文件位置：

- `docs/notes/numerical_stability_summary.md`
- `validation/outputs/archive/normal_state/fs_adaptive_integration/final_check_summary.md`
- `validation/outputs/archive/casimir/local_response_integral/refined_convergence/refined_convergence_summary.md`

推荐阅读顺序：

1. `docs/notes/numerical_stability_summary.md`
2. `validation/outputs/archive/normal_state/fs_adaptive_integration/final_check_summary.md`
3. `validation/outputs/archive/casimir/local_response_integral/refined_convergence/refined_convergence_summary.md`

大型 `.npz`、`.csv`、`.png` 文件是可复现数据和绘图材料，不是主要阅读入口。

本阶段不使用 `.gitignore` 隐藏结果，也不删除旧 outputs。历史输出保留用于追溯数值
判断的来源。

后续 local-response distance scan 应另开目录：

```text
outputs/casimir/local_response_distance_scan/
```

该后续阶段仍应明确标注：local-response benchmark 不是正式 finite momentum Casimir 结论。
