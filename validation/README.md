# Validation

`validation/` 保存能够直接消费当前 two-band finite-q contract 的数值验证入口、可测试实现和轻量证据。根目录不再放置独立 runner 或 analyzer。

## 统一入口

所有 active validation 命令通过：

```bash
python -m validation --help
python -m validation <group> <command> [options]
```

当前分组：

- `ward`：finite-q Ward、collective phase-column 与 phase-Hessian 诊断；
- `static`：exact zero-Matsubara k-grid、quadrature 和 d-wave reference 扫描；
- `matsubara`：正 Matsubara 单点与 `xi -> 0+` 连续性扫描。

示例：

```bash
python -m validation static nk-scan --nks 8 12 16
python -m validation matsubara positive-point --help
python -m validation ward commensurate --help
python -m validation ward phase-column --help
python -m validation ward phase-hessian --help
```

## 目录边界

- `commands/`：CLI 参数、文件读写和终端输出；不承载新的物理公式；
- `lib/`：可测试的数值算法、诊断分解和 validation model adapters；
- `outputs/`：每个验证对象的复现命令、README 和轻量 summary；
- `reports/`：跨验证对象的当前状态与 artifact policy；
- `__main__.py`：稳定的公共命令路由。

新增诊断时，应优先扩展现有 group，而不是在 `validation/` 根目录新增 `run_*.py`、`analyze_*.py` 或 `average_*.py`。

## 输出约定

- 完整 CSV、JSON 和 log 写入各主题的 `raw/`，由 Git 忽略；
- Git 中只保留 README、command、summary 和小型 status；
- 被新诊断替代的历史输出说明应删除，不建立 archive 目录；
- 正式主计算结果进入根目录 `outputs/`；
- 不使用根目录 `results/`。

当前 validation 只证明相应数值门禁；在 exact-static reference、正频连续性和外层积分报告全部完成前，不宣称 finite-q Casimir production-ready。
