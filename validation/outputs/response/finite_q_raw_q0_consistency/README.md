# finite-q raw q=0 formula consistency 诊断输出

本目录保存 raw q=0 finite-q bubble 与 local response 定义层级的一致性诊断结果。

这里的 raw q=0 bubble 会绕过 q=0 local-reference hook，强制使用与 q>0 相同的
finite-q bubble 公式。诊断目标是判断它更接近 local_sigma、K_para、K_total/omega
还是无法匹配任何已有 local component。

这些输出只属于 response 层公式诊断，不是 Lifshitz / Casimir 输入，不是 torque 结果，
也不是正式物理结论。

限制：
- gauge_status=prototype_not_ward_verified
- finite-q diamagnetic / Ward closure 未完成
- n=0 model 未完成
- final_casimir_input=False
- not_final_Casimir_conclusion=True
