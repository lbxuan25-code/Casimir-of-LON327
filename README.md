# Casimir-of-LON327

本仓库实现 LNO327 minimal model 的 finite-q response、微观点认证与完整自适应 Casimir 外积分。

## 唯一主计算路线

```text
microscopic finite-q certification
→ radial adaptivity
→ angular adaptivity + offset audit
→ joint radial/angular error budget
→ adaptive outer-Q cutoff + tail bound
→ adaptive Matsubara cutoff + tail bound
```

稳定公共入口只有：

```python
from lno327.casimir import build_full_casimir_config, run_full_casimir

config = build_full_casimir_config(
    pairings=("spm",),
    temperature_K=10.0,
    separation_nm=20.0,
    plate_angles_deg=(0.0, 17.0),
)
result = run_full_casimir(config)
```

命令行入口：

```bash
python -m lno327.casimir \
  --case spm_T10K_d20nm_theta17deg \
  --pairings spm \
  --temperature-K 10 \
  --separation-nm 20 \
  --plate-angles-deg 0 17
```

运行产物统一进入：

```text
outputs/casimir/runs/<case>/
```

其中 `<case>` 是人为命名的物理算例，不使用开发版本号命名。

## 物理边界

外积分控制结构已经完整，但真实 LNO327 全栈资格验证尚未完成。因此结果仍保持 fail-closed：

```text
production_casimir_allowed = false
```

这表示仓库和运行路线已生产化，不表示当前响应模型已经获得最终物理授权。

## 固定网格历史路线

旧固定网格控制器不再从包顶层导出。它只保留作回归参考：

```python
from lno327.casimir.legacy import run_fixed_reference_casimir
```

新计算不得以该接口作为主路线。

## 目录

- `src/lno327/casimir/`：主计算及数值合同；
- `validation/`：独立诊断、资格检验与不可变参考证据；
- `tests/`：稳定数值合同与全栈 fail-closed 测试；
- `docs/casimir/`：主路线、误差预算、运行和维护说明；
- `outputs/`：本地运行产物布局说明，生成数据不提交；
- `scripts/`：非 Casimir 主入口的研究辅助脚本。

## 检查

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m lno327.casimir --help
```
