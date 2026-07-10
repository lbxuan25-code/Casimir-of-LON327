# Validation

`validation/` 只保存能够直接消费当前 two-band finite-q production contract 的数值验证入口和轻量证据。

## 当前 active surface

- `run_static_nk_scan.py`：exact zero-Matsubara 固定非零 q 的 k-grid 收敛与耗时扫描；
- `lib/finite_q_validation_models.py`：仅暴露 `symmetry_bdg_2band`；
- `outputs/zero_matsubara/static_nk_convergence/`：当前扫描说明、复现命令和本地 raw 输出位置；
- `reports/`：当前状态和 artifact policy。

历史 stage 脚本、四轨道诊断、local-q=0 零模代理、旧 TE/TM adapter、旧 Ward triage 和 sandbox 路线已经从当前工作树删除。需要追溯时使用 Git 历史，而不是在 active validation 中并行维护。

## 输出约定

- 完整 CSV、JSON 和 log 写入各主题的 `raw/`，由 Git 忽略；
- Git 中只保留 README、command、summary 和小型 status；
- 正式主计算结果进入根目录 `outputs/`；
- 不使用根目录 `results/`。

当前 validation 只证明相应数值门禁；在收敛报告完成前，不宣称 finite-q Casimir production-ready。
