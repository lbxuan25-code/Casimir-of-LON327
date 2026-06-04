# response 输出入口

当前分支不保留有限动量 response 输出。response 验证输出只保留 local q=0
response、static policy、单位转换和 numerical stability 相关结果。

当前 local-response / Casimir benchmark 的主要阅读入口：

- `validation/outputs/response/bdg_static_gauge_closure/`
- `outputs/casimir/local_response_distance_scan/`
- `validation/outputs/archive/response/local_sheet_imag/`
- `validation/outputs/archive/response/static_policy_comparison/`
- `validation/outputs/archive/response/unit_audit/`

`bdg_static_gauge_closure/` 检查 local BdG `K_dia - K_para` 的静态 stiffness、
候选 `rho_s` 的有限性和 C4/offdiag 行为；它不是最终 optical conductivity，
也不是最终 Casimir 输入。

如未来需要重启有限动量 response，应重新设计闭合的 response 层，并在新的目录下重新生成
脚本、测试和输出。
