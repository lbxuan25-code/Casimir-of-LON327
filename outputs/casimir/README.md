# Casimir output layout

`outputs/casimir/` 是本地生成数据根目录。生成数据不提交到 Git；仓库只跟踪本布局说明和生成、审计、归档代码。

## 正式根级目录

```text
outputs/casimir/
├── README.md
├── runs/             # 当前解压运行及其运行内 reports
├── archive/          # 已验证冷归档；legacy 子层只用于历史产物
├── catalog/          # registry、catalog、plan、verification、execution
├── reports/          # 当前全局报告及经过验证的 compact/sidecar
├── workflow_logs/    # 当前后台工作流 PID、命令、状态和日志
└── postprocessed/    # 可选后处理输出
```

`workflow_logs/` 是当前生产工作流的正式路径，不属于历史垃圾。`postprocessed/` 可以在首次后处理前不存在。

根级 `diagnostics/` 是过渡性目录：新的单次运行诊断应写入 `runs/<case>/reports/diagnostics.json`。在完成内容和引用审计前，不得直接删除或迁移根级 `diagnostics/`。

下列旧路径不属于最终根级布局，应先通过只读审计确认内容、引用和归档安全性，再迁入 `archive/legacy/`：

```text
0deg_runtime_budget_pilot_logs/
N896_scan_logs/
0deg_pilot_v2_diagnostics.tar.gz
dwave_0deg_pilot_cache.tar.gz
```

运行只读布局审计：

```bash
python -m scripts.full_casimir.data layout-audit
```

默认写入：

```text
outputs/casimir/catalog/output_layout_audit.json
outputs/casimir/catalog/output_layout_audit.tsv
```

审计会记录每个根级条目的分类、大小、文件数、目录摘要、旧 tar 成员结构、JSON schema 和仓库文本引用。该命令不会移动、覆盖或删除任何输出。

## 正式运行

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

`audit` 子命令对多个 pairing 使用同一审计结构：逐点重放候选全局 `logdet_rtol`、比较 pairing-blind 数值政策、在临时缓存中投影候选策略、重放同一套 radial/angular/outer/Matsubara 控制器，并生成一份统一报告。

```bash
python -m scripts.full_casimir.diagnostics audit \
  --run-dir outputs/casimir/runs/dwave_T10K_d20nm_theta_p000deg_0deg_pilot_v4 \
  --run-dir outputs/casimir/runs/spm_T10K_d20nm_theta_p000deg_0deg_pilot_v4 \
  --candidate-logdet-rtol 0.0015 0.00175 0.002 0.0025 0.003 \
  --closure-candidate-logdet-rtol 0.002 0.0025 \
  --unified-radial-budget-fraction 0.8
```

完整审计包括：

- 原生产策略下逐点 `status/working_N/audit_N/establishment_mode` 等价性；
- 统一 N 梯子及统一 radial/angular 预算下的临时候选缓存重放；
- 从最终接受的 child-panel 求积网格重建 exact-q signed/absolute 权重；
- 候选策略相对最高已存硬物理层的加权自由能变化与保守微观点误差界；
- 按加权误差贡献生成独立高 N holdout 清单；
- pairing-independent 的条件性真空传播尾界；
- 避免重复累加 radial/angular/offset 子误差的端到端误差账本；
- cache-only 控制器重放耗时及 radial/angular 预算反事实筛选。

默认全局报告写入 `outputs/casimir/reports/convergence_audit.json`；经过验证的 compact/sidecar 表示可以替代原始大型 JSON。所有原缓存运行前后 SHA-256 必须相同，任何 cache miss 都禁止启动新微观计算。

审计实现完成不等于生产参数已经获准。报告在以下外部证据完成前必须保持 `production_change_not_authorized`：

1. 按冻结候选执行独立高 N holdout；
2. 证明当前功率度量下 round-trip reflection operator 的收缩性，从而把条件性解析尾界升级为正式认证；
3. 使用真实微观计算比较冻结候选与严格策略的 wall time、内存和缓存行为；
4. 将源生产配置改为 pairing-blind 数值政策并重新进行 0° 端到端认证。

条件性解析尾界只报告公式、数值和缺失前提，不能单独放行 outer tail；点级 `N^2` 与 cache-only replay wall time 也不能冒充新生产运行的实际加速。
