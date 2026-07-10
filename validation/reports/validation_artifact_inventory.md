# Validation artifact policy

## 长期保留

- active validation Python 入口；
- 当前模型 adapter；
- README、command、summary、status；
- 小型 machine-readable convergence 摘要。

## 不长期保留

- raw CSV/JSON/log；
- `.npz`、`.npy`、cache tensors；
- intermediate outputs；
- repeated figures；
- 已被 production contract 取代的历史诊断脚本和输出。

完整扫描写入 `validation/outputs/<topic>/raw/` 并由 Git 忽略。正式结果只进入根目录 `outputs/`。历史实现通过 Git history 追溯，不在当前工作树保留 archive 或 sandbox 副本。
