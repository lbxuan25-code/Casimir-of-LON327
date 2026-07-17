# 文档入口

本目录保存理论主线、稳定数值合同和运行维护说明，不保存开发流水或生成结果。

推荐阅读顺序：

1. `current_route.md`：当前唯一计算路线和物理状态；
2. `casimir/README.md`：完整自适应 Casimir 主入口；
3. `casimir/numerical_contract.md`：径向、角向和双尾误差合同；
4. `casimir/operations.md`：正式运行、恢复与产物读取；
5. `theory_path.md`：理论计算路径；
6. `implementation_design.md`：代码对象与理论对象的对应；
7. `references/`：论文和材料背景。

独立资格检验见 `../validation/`，生成产物布局见 `../outputs/`。Casimir 正式入口只见顶层 `../README.md`，不通过 `scripts/` 启动。
