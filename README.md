# Casimir-of-LON327

## 项目定位

本仓库围绕 LNO327 minimal model，建立 normal / BdG response、finite-q response、unit / reflection input 与 Casimir benchmark 的研究型计算框架。

当前仓库不是最终材料结论仓库，也不输出最终 Casimir torque、force 或 energy 结论。

## 当前状态

- local `q=0` response baseline 已形成；
- validation 输出已归档为 summary / status / command；
- finite-q BdG response 已完成架构解耦；
- generic finite-q engine 与 `PairingAnsatz` 输入层已分离；
- finite-q Ward / gauge closure 尚未完成；
- raw finite-q BdG response 不能作为正式 Casimir input；
- 当前没有最终 Casimir torque、force 或 energy 结论。

## 阅读入口

- `docs/README.md`：理论主线与工程设计入口；
- `docs/current_route.md`：当前总路线进行到哪一步；
- `docs/theory_path.md`：理论计算路径；
- `docs/implementation_design.md`：代码工程如何对应理论对象；
- `docs/references/`：参考文献与背景资料；
- `validation/README.md`：数值检验、status、summary 和复现命令；
- `outputs/README.md`：当前主计算产物；
- `scripts/README.md`：可运行脚本入口。

## 快速检查

```bash
pytest
```

最小导入入口：

```python
from lno327.api import KuboConfig, PairingAmplitudes, local_response_imag_axis
```

## 目录结构

- `src/lno327/`：核心计算实现；
- `scripts/`：运行入口；
- `outputs/`：主计算产物；
- `validation/`：数值检验证据；
- `docs/`：理论主线与工程设计；
- `docs/references/`：参考文献和背景资料。
