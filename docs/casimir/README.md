# Casimir production route

## Public surface

```python
from lno327.casimir import (
    FullCasimirConfig,
    FullCasimirResult,
    build_full_casimir_config,
    run_full_casimir,
)
```

`run_full_casimir` 是唯一顶层计算路线。各级 adaptive controller 是实现模块和数值开发面，不从包顶层构成竞争入口。

## Layer order

```text
FrequencyExtendableCertifiedOuterQProvider
→ adaptive radial panels
→ global periodic angular order and offset audit
→ joint radial/angular direction selection
→ cumulative u cutoff and outer tail envelope
→ cumulative Matsubara cutoff and high-frequency tail envelope
```

所有层逐 pairing、逐 Matsubara channel 保留误差证据。任何 microscopic point、有限域积分或 tail 证据未建立时，结果均返回 `unresolved`。

## Documents

- `numerical_contract.md`：误差预算和 tail 规则；
- `operations.md`：运行、恢复、输出和首个 pilot；
- `legacy_fixed_reference.md`：固定网格历史路线的隔离边界。
