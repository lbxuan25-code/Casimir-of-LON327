# 论文输出整理指南

本说明用于把仓库输出逐步整理成论文草稿可用的图、表和文字材料。当前目标不是宣布
Casimir torque 结论，而是建立可复现、边界清楚的 response 诊断链。

## 图像规范

新版绘图脚本采用统一的 publication style：

- 默认保存 300 dpi PNG。
- 图中使用克制的坐标网格、统一字号和适合论文草稿的紧凑留白。
- 图题保持诊断语义，论文正文中可改写为更短的 panel caption。

若一张图要进入论文草稿，优先使用脚本生成的 300 dpi `.png`。需要重画或复查逐点数据时，
用对应脚本的显式 expanded-data 选项在本地重新生成 CSV/NPZ；大型 `.npz` 不默认进入 GitHub。

## 输出分级

**可作为论文草稿素材**

- `outputs/pairing/gap_structure/`：展示 minimal pairing 的 FS 投影 gap 结构。
- `outputs/normal_state/`：normal-state conductivity baseline。
- `outputs/bdg/*kernel*/`：展示 BdG response 层次，尤其是
  $K_{\mathrm{para}}$、$K_{\mathrm{dia}}$、$K_{\mathrm{total}}$ 的关系。
- `outputs/bdg/superconducting_response_imag/`：展示 $n\ge 1$ 的
  $\Sigma_{\mathrm{SC}}(i\xi_n)$。

**只作为方法边界或附录说明**

- `docs/notes/casimir_torque_response_pipeline_zh.md`：说明 finite-q response 到未来 reflection / Casimir 的当前总览路线和禁止事项。
- `validation/outputs/response/normal_finite_q_kernel_convergence/`：可作为 Stage 1
  normal finite-q current-current kernel same-interface 收敛的验证附录素材；不能作为
  finite-q conductivity 或 Casimir 结果图。
- $n=0$ policy 与单位转换目前作为方法边界说明，不作为材料物理结论。

**不作为论文物理结论**

- `outputs/casimir/local_response_distance_scan/`：当前 local-response Casimir 初级结论；
  必须同时说明 `n0_policy=skip` 与 finite-momentum response 未包含。

## Validation / outputs 文件格式规范

`validation/outputs/` 下的输出默认分成 GitHub tracked summary artifacts 和本地可复现的
expanded data。这个规范适用于后续所有 validation outputs，包括 `units`、`response`、
`numerical_stability`、`casimir`、`n0_policy` 等目录。

GitHub tracked outputs 应包含：

- `summary.md` 或同等命名的 Markdown summary；
- `data/*_compact.csv` 或 `data/*_compact_summary.csv`；
- `figures/*.png` publication figures。

完整 expanded CSV/NPZ 属于 reproducible local data，默认不上传 GitHub。若需要逐点完整数据，
使用对应脚本的显式选项重新生成，例如：

```bash
python validation/scripts/units/audit_casimir_q_grid_to_model_q.py --write-expanded-data
```

compact CSV 必须包含足够的聚合指标，能支撑 summary 中的主要结论。figures 只用于展示趋势，
不能替代 compact CSV 中的数值记录。大型 CSV/NPZ 不进入 GitHub，避免触发 50 MB warning
或 100 MB hard limit；不为 validation outputs 引入 Git LFS。

## 推荐论文叙事顺序

1. 先给出 normal-state Hamiltonian 与 two minimal pairing ansatz。
2. 展示 gap projection / near-node 结果，说明 sign 仍是 gauge-dependent preliminary
   diagnostic。
3. 展示 normal-state 与 BdG response 的 local $q=0$ 对称性诊断。
4. 说明当前 local isotropic baseline 不产生可分辨面内各向异性，因此不能从当前数据
   声称 finite Casimir torque。
5. 明确 $n=0$、finite-$q_{\parallel}$ 和真实各向异性机制是未来正式 Casimir 阶段的
   必要前提。

## n=0 文字模板

可在报告或论文草稿中使用如下保守表述：

> Lifshitz 求和形式上包含半权重的 $n=0$ Matsubara 项。但在当前局域各向同性
> baseline 中，超导响应 $\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega$
> 只定义于 $n\ge 1$。因此 baseline 计算省略 $n=0$ 贡献，以避免引入未定义的
> 零频 sheet conductivity。lowest-Matsubara extrapolation 与 static-kernel
> 分支只作为诊断输出，不用于最终 Casimir torque 结论。

## 重画图像建议

重新生成论文风格图时，优先使用固定参数并在文件名前保留参数线索，例如：

```bash
python validation/scripts/response/compare_local_sheet_response_imag.py \
  --kinds normal spm dwave \
  --delta0 0.04 --nk 24 --temperature 30 \
  --matsubara-min 1 --matsubara-max 8 \
  --eta 0.0001 \
  --output-prefix validation/outputs/response/local_sheet_imag/data/local_sheet_response_imag
```

生成后检查 compact CSV、summary 和 `.png` 图像是否存在；只有显式请求 expanded data 时，
才检查本地 `.npz`/expanded CSV。
