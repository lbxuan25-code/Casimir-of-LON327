# Validation Guide

本目录集中保存数值可信度、收敛性、公式诊断、Casimir benchmark、历史 smoke 和 cache。
主 `scripts/` / `outputs/` 现在只保留材料和模型本征特性的当前计算入口与结果。

## 目录结构

- `scripts/numerical_stability/`：response convergence、normal sampling、high-Nk、n=0 sensitivity 等。
- `scripts/response/`：local sheet response、static policy。
- `scripts/casimir/`：local-response Casimir benchmark、distance scan、convergence runner。
- `scripts/smoke/`：历史 smoke / plumbing 检查。
- `scripts/compat/`：迁移后的旧命令兼容入口。
- `outputs/`：上述验证脚本对应的输出、历史归档和 summary。
- `cache/`：可复用中间张量，例如 local-response Casimir response tensors。

## 阅读顺序

1. `../docs/reports/current_project_status.md`
2. `outputs/numerical_stability/README.md`
3. `outputs/casimir/local_response_integral/distance_scan/distance_scan_summary.md`
4. `outputs/archive/ARCHIVE_INDEX.md`

## 维护原则

- validation 结果是支撑证据，不是主 `outputs/` 的材料本征结果。
- 大型 `.csv`、`.npz`、`.png` 继续保留并上传，便于 ChatGPT 或外部审阅复查。
- 新的 convergence / diagnostic / benchmark-only 输出默认写入 `validation/outputs/`。
- 新的 cache 默认写入 `validation/cache/`。
