# Validation Guide

本目录集中保存数值可信度、收敛性、公式诊断、Casimir convergence benchmark、
历史 smoke 和 cache。主 `scripts/` / `outputs/` 保存当前计算入口、材料结果和边界
清楚的初级结论。

## 目录结构

- `scripts/numerical_stability/`：response convergence、normal sampling、high-Nk、n=0 sensitivity 等。
- `scripts/response/`：local sheet response、static policy。
- `scripts/casimir/`：local-response Casimir convergence、refinement 和流程检查。
- `scripts/smoke/`：历史 smoke / plumbing 检查。
- `outputs/`：上述验证脚本对应的当前输出和 summary。
- `cache/`：可复用中间张量，例如 local-response Casimir response tensors。

## 阅读顺序

1. `../docs/reports/current_project_status.md`
2. `outputs/numerical_stability/README.md`
3. `../outputs/casimir/local_response_distance_scan/distance_scan_summary.md`
4. `outputs/response/normal_finite_q_kernel_convergence/normal_finite_q_kernel_convergence_summary.md`

## 维护原则

- validation 结果是支撑证据，不是主 `outputs/` 的材料本征结果。
- 大型 `.csv`、`.npz`、`.png` 继续保留并上传，便于 ChatGPT 或外部审阅复查。
- `.csv` 作为可读表格摘要；同名 `.npz` 可以额外包含运行参数、网格列表、
  partial sums 或其他中间数组，因此不要求两者字段完全一一对应。
- 新的 convergence / diagnostic / benchmark-only 输出默认写入 `validation/outputs/`。
- 新的 cache 默认写入 `validation/cache/`。
