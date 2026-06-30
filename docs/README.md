# 文档入口

本目录保存仓库的理论主线和工程设计说明。

它不是：

- validation 结果目录；
- 运行命令手册；
- 开发流水记录；
- outputs 数据说明。

推荐阅读顺序：

1. `current_route.md`：当前总路线进行到了哪一步；
2. `theory_path.md`：从物理目标到 Casimir observable 的理论计算路径；
3. `implementation_design.md`：代码工程如何对应理论对象；
4. `finite_q_diagnostic_pipeline.md`：finite-q Ward diagnostic 的当前流水线和边界；
5. `best_effort_finite_q_casimir_route.md`：Ward 未闭合前用于跑通下游 plumbing 的 best-effort finite-q 路线；
6. `references/`：论文、理论背景和材料参考。

详细数值检验见 `../validation/`。
主计算产物见 `../outputs/`。
运行入口见顶层 `../README.md` 和 `../scripts/`。
