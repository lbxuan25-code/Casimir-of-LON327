# Operations

## Named run

```bash
python -m lno327.casimir \
  --case spm_T10K_d20nm_theta17deg \
  --pairings spm \
  --temperature-K 10 \
  --separation-nm 20 \
  --plate-angles-deg 0 17 \
  --memory-budget-gb 24
```

算例名必须稳定描述物理输入，不使用版本号或临时状态命名。

## Resume

```bash
python -m lno327.casimir \
  --case spm_T10K_d20nm_theta17deg \
  --resume
```

恢复只复用相同 microscopic policy 下的 `(pairing, n, qx.hex, qy.hex)` 条目；温度、角度、pairing、N ladder 或物理 gate 改变时缓存会 fail-closed。CLI 还会在启动前比较完整 `config.json`，配置不一致时要求使用新的 case 名称。

## Artifact reading order

1. `manifest.json`：运行是否完成及 commit；
2. `summary.json`：选定 cutoff、误差和终止原因；
3. `result.json`：完整逐层审计；
4. `cache/certified_points.json`：恢复数据，不作人工结果解释。

## First real pilot

首个正式点使用人为指定的 `spm, T=10 K, d=20 nm, theta=(0,17 deg)`。先看是否闭合及终止原因，不在首次运行中同时加入 d-wave 或参数扫描。
