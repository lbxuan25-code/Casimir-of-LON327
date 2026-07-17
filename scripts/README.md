# 研究辅助脚本

`scripts/` 只保存 normal-state、pairing、BdG 与模型检查等研究辅助入口。

Casimir 主计算不再通过 `scripts/casimir/` 中的阶段性脚本启动。唯一正式入口是：

```bash
python -m lno327.casimir --help
```

数值资格检验、收敛诊断和 smoke 命令属于 `validation` CLI，不应包装成第二套主计算脚本。
