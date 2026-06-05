# Publication Output Guide

本说明用于把仓库输出逐步整理成论文草稿可用的图、表和文字材料。当前目标不是宣布
Casimir torque 结论，而是建立可复现、边界清楚的 response 诊断链。

## 图像规范

新版绘图脚本采用统一的 publication style：

- 默认保存 300 dpi PNG。
- 图中使用克制的坐标网格、统一字号和适合论文草稿的紧凑留白。
- 图题保持诊断语义，论文正文中可改写为更短的 panel caption。

若一张图要进入论文草稿，优先使用脚本生成的 300 dpi `.png`，并保留对应 `.npz`
数据以便最终阶段按期刊要求重画。

## 输出分级

**可作为论文草稿素材**

- `outputs/pairing/gap_structure/`：展示 minimal pairing 的 FS 投影 gap 结构。
- `outputs/normal_state/`：normal-state conductivity baseline。
- `outputs/bdg/*kernel*/`：展示 BdG response 层次，尤其是
  $K_{\mathrm{para}}$、$K_{\mathrm{dia}}$、$K_{\mathrm{total}}$ 的关系。
- `outputs/bdg/superconducting_response_imag/`：展示 $n\ge 1$ 的
  $\Sigma_{\mathrm{SC}}(i\xi_n)$。
- `validation/outputs/response/normal_finite_q_kernel_convergence/`：展示 Stage 1
  normal finite-q current-current kernel same-interface 收敛诊断。

**只作为方法边界或附录说明**

- `docs/notes/finite_q_response_plan_zh.md`：说明 finite-q 后续阶段和禁止事项。
- $n=0$ policy 与单位转换目前作为方法边界说明，不作为材料物理结论。

**不作为论文物理结论**

- `outputs/casimir/local_response_distance_scan/`：当前 local-response Casimir 初级结论；
  必须同时说明 `n0_policy=skip` 与 finite-momentum response 未包含。

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

> The Lifshitz sum formally contains the half-weighted $n=0$ Matsubara term.
> In the present local isotropic baseline, however, the superconducting response
> $\Sigma_{\mathrm{SC}}=K_{\mathrm{total}}/\omega$ is defined only for
> $n\ge 1$. We therefore omit the $n=0$ contribution in the baseline calculation
> to avoid introducing an undefined zero-frequency sheet conductivity. The
> lowest-Matsubara extrapolation and static-kernel branches are reported only as
> diagnostics and are not used for final Casimir-torque conclusions.

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

生成后检查 `.npz` 数据和 `.png` 图像是否同时存在。
