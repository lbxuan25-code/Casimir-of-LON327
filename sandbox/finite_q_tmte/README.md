# finite_q_tmte sandbox

这是有限 q 直接 TM/TE 目标基响应方法的隔离 sandbox。当前版本只用于工程验证和诊断，不是 Casimir 生产输入。

核心约束：

- 主路径直接构造 `G/TM/TE` 目标基顶点与目标基响应块。
- 主路径不先构造完整 `3x3` 电磁分量响应再旋转或投影。
- `G` 纯规范源只作为诊断；候选物理对象是 Schur 后切出的 `K_TMTE_eff`。
- `valid_for_casimir_input` 在 sandbox v1 中始终为 `false`。
- 生产代码 `src` 不依赖、也不应导入本 sandbox。

典型轻量检查：

```bash
python -m compileall sandbox/finite_q_tmte
PYTHONPATH=src:. pytest sandbox/finite_q_tmte/tests -q
```

示例扫描命令仅供人工后续运行，不应在实现阶段自动执行：

```bash
PYTHONPATH=src:. python sandbox/finite_q_tmte/scripts/run_scan.py \
  --model symmetry_bdg_2band \
  --pairing dwave \
  --xi 0.01 \
  --q-values 0.02 \
  --nk 13 \
  --shift-fractions 0.0 0.2 0.4 0.6 0.8 \
  --output-dir sandbox/finite_q_tmte/outputs/dwave_q002_nk13
```

