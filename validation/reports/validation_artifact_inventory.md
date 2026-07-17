# Validation artifact policy

## 长期保留

- active validation Python 入口；
- 当前模型 adapter；
- 被多个 active quadrature 复用的中性 workspace 与物理后处理；
- README、command、summary、status；
- 小型 machine-readable convergence 摘要。

## 不长期保留

- raw CSV/JSON/txt/log；
- `.npz`、`.npy`、cache tensors；
- intermediate outputs；
- repeated figures；
- 已否决 quadrature 的 backend、wrapper、CLI、专用测试和专用文档；
- 已被 production contract 取代的历史诊断脚本和输出。

完整扫描是可复现本地产物，由 Git 忽略。正式结果只在 production reference 建立后进入根目录 `outputs/`。历史实现和拒绝证据通过 Git history、当前主合同和 PR 说明追溯，不在当前工作树保留 archive、sandbox 副本或可运行的失败方法入口。
