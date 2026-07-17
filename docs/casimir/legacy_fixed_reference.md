# Legacy fixed-grid reference

固定网格链保留用于：

- qualified `spm, n=0,1` golden reference 回归；
- 固定 quadrature measure、prime weight 和 microscopic certification 的单元合同；
- 与完整自适应结果的诊断比较。

它不属于主计算路线，且不从 `lno327.casimir` 包顶层导出。

```python
from lno327.casimir.legacy import (
    FixedCasimirConfig,
    run_fixed_reference_casimir,
)
```

禁止以固定链结果替代 outer-Q tail 或 Matsubara tail 认证。
