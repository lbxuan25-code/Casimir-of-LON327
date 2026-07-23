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
→ persistent certified MaterialResponseSnapshot
→ reflection / propagation / logdet geometry assembly
→ observable-specific integration
→ observable-specific error and production admission
```

当前边界文档：

```text
docs/casimir/todo2_material_response_boundary.md
docs/casimir/todo3_persistent_response_cache.md
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
- 只保存认证成功响应的持久化、content-addressed response cache；
- pairing、有限温度、Matsubara 频率、exact `q_crystal`、模型、响应政策，以及
  完整 N ladder、ordered exact shifts、canonical reduction policy 的严格缓存身份；
- 持久化 artifact 对 working/audit N 与 audit-shift provenance 的身份一致性检查；
- NPZ + canonical JSON manifest、array checksum、原子写入、只读加载、
  typed failure 和同身份冲突检测；
- `disabled`、`populate` 与 strict `read_only` 三种缓存模式；
- cache hit 直接加载，partial hit 只把缺失频率送入 microscopic engine；
- strict read-only miss 不允许 microscopic fallback；
- 从 live sample 或 persisted snapshot 到单板 reflection 和双板 signed logdet
  的纯几何组装；
- 零频与正频的 live/persisted geometry replay 等价测试；
- repository-level import guards，保持 material、persistence、cache orchestration、
  geometry 和 observable 层的单向依赖；
- 所有新对象保持 diagnostic-only。

仍未建立：

- TODO 4 的完整多距离、多角度批量几何调度与旧逐点流程 qualification；
- TODO 5 的频率压缩与独立 holdout 误差合同；
- TODO 7 的 q/角度 surrogate response library；
- TODO 10 的 observable-level error budget 和生产准入；
- 真实 LNO327 四轨道模型的全栈资格。

## 缓存边界

新的 persistent response cache 只接受
`response_certified_diagnostic` 的材料响应。未认证的 N/shift 中间结果和
`unresolved` 响应不进入 certified library。

材料响应的目标物理身份与 N/shift provenance 仍保持分离；但一个持久化
“认证响应”的身份必须同时说明它是在什么 N ladder、ordered exact shift set 和
canonical reduction policy 下取得认证。请求这些认证政策中的任一项发生变化时，
必须产生 cache miss，不能用已有结果绕过新请求的认证。

缓存身份不包含距离、板角、`q_lab`、outer quadrature、worker 数、runtime
chunk、路径或运行时间。不同的 exact `q_crystal` 仍然是不同响应；禁止通过
最近邻、取整、插值或旋转未在目标 `q_crystal` 上计算的响应来制造 cache hit。

旧的进程内 `CrystalResponseCache` 与历史 point cache 没有被升级、迁移或
重命名为新的 certified response cache。

## 旧路线的地位

`fixed_transverse_point_engine.py` 与
`fixed_transverse_point_certification.py` 仍保存旧的 geometry-specific
`two_plate_logdet` sweet-spot 路线，用于历史回归和归档结果解释。它们不是新的
可复用材料响应架构，也不能把旧 point cache 自动提升为 response cache。

迁移期间禁止静默 fallback：新材料响应链或 strict cache-only 路线失败时，
不允许自动退回旧 logdet 认证链并把结果标成等价。

## 接下来的顺序

```text
TODO 4 fast geometry assembly and direct old/new equivalence qualification
→ TODO 5 finite-temperature frequency compression/reference-subtraction pilot
→ high-frequency convergence of Delta F / torque / pressure
→ observable-specific production qualification
```
