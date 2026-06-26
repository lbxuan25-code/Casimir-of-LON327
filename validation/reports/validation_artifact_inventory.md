# Validation artifact inventory

本文件是 `validation/` 的维护政策，不是逐文件流水账。目标是在保留结论和复现能力的同时，让大型 generated artifacts 退出版本控制。

## 长期保留

保留便于 review 且无需重新跑重任务的轻量证据：

- `validation/README.md`；
- `validation/reports/*.md`；
- topic README；
- summary markdown；
- 小型 status / summary JSON 或 CSV；
- `command.sh` 或复现命令片段；
- validation scripts 和 tests。

小型 machine-readable metrics 可以保留；如果 JSON 变成 raw dump，应拆成小型 status/summary 和 ignored raw artifact。

## 不长期保留

以下内容可再生成，应 ignore 或从 Git 删除：

- `validation/cache/**/*.npz`、`.npy`、`.csv`、`.jsonl`；
- `validation/outputs/**/data/*.npz` 和 `.npy`；
- raw / expanded / large data CSV；
- `raw/`、`intermediate/` 输出目录；
- repeated figures；
- benchmark scratch outputs 和 logs。

图像只在报告明确引用且文字/表格不足以表达时保留；优先放在报告附近或 `docs/assets/validation/`，并说明来源。

## cache 与 outputs 的区别

`validation/cache/` 是复用中间张量，存在目的只是加速本地 validation，始终可再生成。

`validation/outputs/` 保存脚本输出。长期保留的只应是 README、summary、status marker 和 command。`data/`、`figures/`、`raw/`、`intermediate/` 默认是本地 artifact。

## 再生成策略

需要复查 raw artifact 时：

1. 阅读对应 `validation/outputs/**/README.md` 和 summary。
2. 查看同目录 `command.sh`。
3. 运行对应 `validation/scripts/**` 重新生成本地 ignored 输出。
4. 只提交新的 README、summary、status 或 command，除非报告明确说明必须保留大型 artifact。

## 当前组织方式

- response kernel / Ward / BdG：`validation/outputs/response/`
- unit conversion / q-grid mapping：`validation/outputs/units/`
- reflection input / adapter：`validation/outputs/reflection/`
- 跨主题总览：`validation/reports/validation_summary.md`

## cleanup snapshot

从 2026-06-26 起，仓库策略从“保留 bulky validation outputs”改为“保留轻量证据”。已删除或忽略的类别包括 cache tensors、binary arrays、CSV data tables、repeated figures 和 scratch logs。删除的 artifact 可由 summary 中列出的脚本入口重新生成。
