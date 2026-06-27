# BdG finite-q 验证输出

本目录保存 diagnostic-only finite-q BdG validation 的轻量输出和复现入口。这里的结果用于定位 superconducting BdG finite-q Ward closure blocker，不是 formal finite-q Casimir 输入。

## 可提交内容

- 小型 markdown summary；
- 小型 JSON / CSV status 或 summary；
- 复现命令文件；
- 简短人工阅读报告。

## 不应提交内容

- raw arrays；
- 大型 CSV；
- expanded logs；
- cache tensors；
- 可重复生成的中间响应矩阵；
- formal Casimir 输入或 torque 结论。

所有 generated finite-q validation results 都必须保持 `valid_for_casimir_input=False`。若需要复查大型数据，请运行本目录的 `command.sh` 或 `validation/scripts/bdg_finite_q/` 下的脚本，在本地重新生成。
