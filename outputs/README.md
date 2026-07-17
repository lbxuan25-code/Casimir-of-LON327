# 运行产物

`outputs/` 只定义正式运行的本地目录结构。生成结果、缓存、日志和图表均不提交到 Git；需要长期保留的紧凑回归证据进入 `validation/references/`。

## Casimir 算例布局

```text
outputs/casimir/runs/<case>/
├── manifest.json
├── config.json
├── summary.json
├── result.json
└── cache/
    └── certified_points.json
```

- `<case>` 是稳定、可读的物理算例名，例如 `spm_T10K_d20nm_theta17deg`；
- 不以 `v1`、`v2`、`latest`、时间戳或临时调试名组织算例；
- `manifest.json` 记录状态、commit 和文件关系；
- `config.json` 是完整嵌套配置；
- `summary.json` 是人工阅读入口；
- `result.json` 保存完整审计证据；
- `cache/` 支持相同物理策略下的恢复和 Matsubara 增量扩展。

## 运行

```bash
python -m lno327.casimir \
  --case spm_T10K_d20nm_theta17deg \
  --pairings spm \
  --temperature-K 10 \
  --separation-nm 20 \
  --plate-angles-deg 0 17
```

同一算例继续使用已有微观点缓存时加 `--resume`。物理或数值配置不兼容时 CLI/provider 会拒绝复用，而不是静默覆盖。
