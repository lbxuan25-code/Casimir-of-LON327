# Casimir-of-LON327

## 项目定位

本仓库围绕 LNO327 minimal model，建立从 normal / BdG response 到 finite-q response、unit / reflection input 与 Casimir benchmark 的研究型计算框架。

local `q=0` response 是当前 baseline；finite-q response 与 Ward / gauge validation 是当前主线中的核心环节。仓库当前不输出最终 Casimir torque、force 或 energy 结论。

## 计算主线

```text
H0(k)
-> pairing ansatz
-> BdG / normal response
-> finite-q response
-> Ward / gauge validation
-> unit conversion
-> reflection input
-> Casimir benchmark
```

## 当前主线位置

- normal / local response 已作为 baseline；
- finite-q BdG response engine 是当前 response 主线；
- `PairingAnsatz` 负责 pairing-dependent 输入，generic finite-q engine 负责通用 response 计算；
- Ward / gauge closure 是 finite-q response 进入 formal conductivity 的关键条件；
- unit conversion、reflection input 与 Matsubara `n=0` policy 是进入 formal Casimir input 的 gating chain；
- 当前 Casimir 相关结果只能作为 benchmark / candidate / baseline，不能作为最终材料结论。

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

## 目录结构

- `src/lno327/`：核心计算实现；
- `scripts/`：当前主计算入口；
- `outputs/`：主计算产物和轻量结果说明；
- `validation/`：数值检验、诊断结果和复现入口；
- `docs/`：理论主线与工程设计；
- `docs/references/`：参考文献和背景资料。
