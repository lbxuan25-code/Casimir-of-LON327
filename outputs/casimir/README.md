# Casimir runs

正式运行只使用 `runs/<case>/`。目录由 `python -m lno327.casimir` 创建；不要手工建立版本化输出树，也不要将生成数据提交到仓库。

## 未收敛 pilot 的安全扩展

`prepare-pilot-extension` 从一个已有 0° pilot 创建新的 profile 缓存种子。源目录不会被修改；目标缓存只保留 `sweet_spot.status == "established"` 的条目，未建立条目会被省略，从而在新运行中定向重算。命令只允许微观点策略完全相同，或目标 `N_candidates` 以完整旧梯子为严格前缀扩展。

针对 d-wave 的局部 N 收敛扩展：

```bash
python -m scripts.full_casimir.workflow prepare-pilot-extension \
  --pairings dwave \
  --source-profile 0deg_pilot_v3 \
  --profile 0deg_pilot_v4 \
  --N-candidates 128 192 256 384 512 640 768 896 1024 1152 1280

python -m scripts.full_casimir.workflow pilots \
  --pairings dwave \
  --profile 0deg_pilot_v4 \
  --N-candidates 128 192 256 384 512 640 768 896 1024 1152 1280
```

针对 SPM 的外层截断扩展，微观点策略不变，因此已建立缓存可全部复用：

```bash
python -m scripts.full_casimir.workflow prepare-pilot-extension \
  --pairings spm \
  --source-profile 0deg_pilot_v3 \
  --profile 0deg_pilot_v4 \
  --outer-cutoffs-u 6 10 14 18 24 30 36 42 48 54 60

python -m scripts.full_casimir.workflow pilots \
  --pairings spm \
  --profile 0deg_pilot_v4 \
  --outer-cutoffs-u 6 10 14 18 24 30 36 42 48 54 60
```

每个目标缓存旁都会写入 `extension_report.json`，记录源/目标 SHA-256、N 梯子、保留条目和被剔除的未收敛点身份。目标缓存已存在时命令只验证策略并跳过，不会覆盖正在形成的结果。

## 未收敛运行诊断

`scripts.full_casimir.diagnostics` 提供只读诊断，不修改原始 `config.json`、`result.json` 或认证缓存。默认模式从缓存中恢复精确 float64 `q`，逐层报告 hard-physical、cross-shift、adjacent-N、连续通过计数和 oscillatory-envelope 门。

```bash
python -m scripts.full_casimir.diagnostics \
  --run-dir outputs/casimir/runs/dwave_T10K_d20nm_theta_p000deg_0deg_pilot_v4
```

当缓存中所有微观点都已经建立时，可以启用 cache-only outer-tail replay。命令先复制认证缓存到临时目录，禁止任何新微观认证工作，然后按原始配置重放外层控制器，输出每个 shell 的 envelope、相邻 shell 比值、有限域误差、尾部预算及主导失败原因。原缓存运行前后 SHA-256 必须一致。

```bash
python -m scripts.full_casimir.diagnostics \
  --run-dir outputs/casimir/runs/spm_T10K_d20nm_theta_p000deg_0deg_pilot_v4 \
  --replay-outer-tail
```

结构化报告写入每个运行目录的 `reports/diagnostics.json`。如果 replay 遇到缓存缺失点，命令会 fail closed，而不会启动昂贵的横向积分。

## 统一收敛审计

`audit` 子命令对多个 pairing 使用同一审计结构：逐点重放候选全局 `logdet_rtol`、比较 pairing-blind 数值政策、在可重放的运行上分离真实 outer shell 信号与有限域误差地板，并生成一份统一的证据账本。

```bash
python -m scripts.full_casimir.diagnostics audit \
  --run-dir outputs/casimir/runs/dwave_T10K_d20nm_theta_p000deg_0deg_pilot_v4 \
  --run-dir outputs/casimir/runs/spm_T10K_d20nm_theta_p000deg_0deg_pilot_v4 \
  --candidate-logdet-rtol 0.0015 0.00175 0.002 0.0025 0.003
```

默认报告写入 `outputs/casimir/reports/convergence_audit.json`。审计不会修改任何认证门：hard-physical gate 与连续通过次数保持不变。点级 `N^2` 只作为成本代理，不冒充 wall time；在缺少求积权重、高 N holdout、解析尾界或无重复计数的总误差账本时，报告必须保持 `production_change_not_authorized`。
