# Validation

`validation/` 只保存能够直接消费当前 two-band finite-q production contract 的数值验证入口和轻量证据。

## 当前 active surface

- `run_static_nk_scan.py`：通用 exact zero-Matsubara 固定非零 q 的 k-grid 基线扫描；
- `run_dwave_static_shift_batch_scan.py`：固定 base grid 上的 nested complete-periodic shift-prefix 扫描；
- `run_dwave_static_shift_ensemble_reference_scan.py`：固定四-shift 规则的跨-`nk` reference 与 cross-rule 门禁；
- `run_dwave_static_shift_budget_scan.py`：等总积分点的 4/8/16-shift 预算分配诊断；
- `run_dwave_small_xi_extrapolation_scan.py`：在外部静态 reference 收敛后进行 `xi -> 0+` 连续性比较；
- `run_positive_matsubara_point_scan.py`：正 Matsubara 单点链条与 k-grid 收敛；
- `lib/finite_q_validation_models.py`：仅暴露 `symmetry_bdg_2band`；
- `reports/`：当前状态和 artifact policy。

已被否定或替代的 nodal adaptive、vector adaptive、旧 periodic-multishift、单-midpoint reference、空间定位、谱分类、band-pair 与 signed-reconstruction 临时入口从当前工作树删除。结论需要追溯时使用 Git 历史，不在 active validation 中平行维护历史 runner。

## 输出约定

- 完整 CSV、JSON 和 log 写入各主题的 `raw/`，由 Git 忽略；
- Git 中只保留 README、command、summary 和小型 status；
- 被新诊断替代的历史输出说明也应删除，不建立 archive 目录；
- 正式主计算结果进入根目录 `outputs/`；
- 不使用根目录 `results/`。

当前 validation 只证明相应数值门禁；在 exact-static reference、正频连续性和外层积分报告全部完成前，不宣称 finite-q Casimir production-ready。
