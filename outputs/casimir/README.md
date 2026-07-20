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
