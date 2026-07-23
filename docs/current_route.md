# 当前总路线

## 当前结论

旧的零角度 absolute-free-energy campaign 已停止并归档，不允许继续补算
`n=32..63`，也不能作为新生产扫描的 seed。当前工作的首要目标已经从
“继续扩大旧全栈 pilot”转为降低单次材料响应成本，并使同一有限温度材料响应
能够被多个角度、距离和 observable 重复使用。

`production_casimir_allowed` 继续为 `false`。

## 新主计算链

```text
H0(k) + pairing ansatz
→ finite-q microscopic integration at (T, xi, q_crystal)
→ geometry-independent MaterialResponseSample
→ response-space N/shift certification
→ reusable response library (TODO 3)
→ reflection / propagation / logdet geometry assembly
→ observable-specific integration
→ observable-specific error and production admission
```

TODO 2 的实现与边界记录在：

```text
docs/casimir/todo2_material_response_boundary.md
```

## 当前实现状态

已经建立：

- 几何无关的零频与正频材料响应合同；
- 包含静态与正频全部 validation tolerances 的 canonical material policy；
- 使用 exact-float frequency/q、material state、policy 与 basis provenance 的
  material identity；
- `q_crystal` 基底中的 response-level N/shift 收敛判断；
- 完整 pairwise cross-shift 与多 N/shift envelope；
- 不接受角度、距离或 outer quadrature 的材料响应 ladder engine；
- 从预计算材料响应到单板 reflection 和双板 signed logdet 的纯几何组装；
- 零频和正频的新旧公式路径等价测试；
- 所有新对象保持 diagnostic-only。

仍未建立：

- TODO 3 的持久化、原子化、可复用 response cache；
- TODO 5 的频率压缩与 holdout 误差合同；
- TODO 10 的 observable-level error budget 和生产准入；
- 真实 LNO327 四轨道模型的全栈资格。

## 旧路线的地位

`fixed_transverse_point_engine.py` 与
`fixed_transverse_point_certification.py` 仍保存旧的 geometry-specific
`two_plate_logdet` sweet-spot 路线，用于历史回归和归档结果解释。它们不是新的
可复用材料响应架构，也不能把旧 point cache 自动提升为 response cache。

迁移期间禁止静默 fallback：新材料响应链失败时，不允许自动退回旧 logdet
认证链并把结果标成等价。

## 接下来的顺序

```text
TODO 2 边界与回归测试闭合
→ TODO 3 reusable response cache
→ 小型 frequency-compression/reference-subtraction pilot
→ high-frequency convergence of Delta F / torque / pressure
→ observable-specific production qualification
```
