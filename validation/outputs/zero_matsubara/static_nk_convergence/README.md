# Zero-Matsubara static k-grid convergence

本目录对应 exact `xi=0`、固定非零 q 的 two-band static response 收敛扫描。

当前尚未提交本地收敛结论。运行 `command.sh` 后，完整 CSV、JSON 和 log 写入 `raw/`；该目录由 Git 忽略。完成扫描后，只提交人工审阅过的 `summary.md` 或小型 status 文件。

主要观测量：

- primitive/effective Ward residual；
- Schur condition number；
- imaginary residual；
- longitudinal gauge leakage；
- density-transverse mixing；
- `chi_bar` 与 `Dbar_T`；
- material cache、q workspace、response 和后处理耗时；
- peak RSS。
