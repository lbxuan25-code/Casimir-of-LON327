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
→ exact read-only geometry plan / prepared reflection / multi-distance logdet
→ observable-specific integration
→ observable-specific error and production admission
```

当前边界文档：

```text
docs/casimir/todo2_material_response_boundary.md
docs/casimir/todo3_persistent_response_cache.md
docs/casimir/todo4_fast_geometry_assembly.md
docs/casimir/todo4_representative_qualification_runbook.md
docs/casimir/todo4_observable_impact_calibration.md
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
- 独立的 response-cache request identity 边界，使 geometry plan 不依赖 cached engine；
- 从 live sample 或 persisted snapshot 到单板 reflection 和双板 signed logdet
  的纯几何组装；
- exact `q_crystal = R(-theta_plate) @ q_lab` 的 geometry batch plan；
- 对 unique exact response identities 的 strict read-only 全量 preflight；
- 每个 response/q/angle 只构造一次 reflection，每个两板组合只构造一次
  distance-independent `R1 @ R2`；
- 多距离只更新传播因子和 signed logdet 的 prepared-pair evaluator；
- zero/positive Matsubara 的 scalar/batch geometry replay；
- 隔离的 scalar-vs-batch qualification；
- 从 certified artifact 读取 working N 与 primary shift、只执行一个旧 microscopic
  point 的 narrow legacy replay runner；
- 旧/新比较对 individual reflections、product、eigenvalues、signed logdet、exact q，
  以及 material/primitive/phase/N/shift/reduction 合同的硬检查；
- 使用分单位 absolute tolerances 的 reduced fixed-outer replay contract；
- frozen representative manifest 与 validation-only staged runner；
- unresolved exact misses 的只读诊断阶段，保存逐 N、逐 shift 的 hard physical、
  cross-shift、adjacent-N 与 complete envelope 证据，且不写 certified cache；
- unresolved diagnostics 到 local observable 的独立 calibration：使用 exact TODO 4
  reflection/logdet 几何，枚举两块板全部 `3 x 3` shift pairs，记录平行 control、
  finite-angle logdet contrast、round-trip eigenvalues、相邻 N 变化和有限 `n=0,1`
  Matsubara-weighted partial sum；该阶段不读写 certified cache，也不提升 unresolved
  response；
- repository-level import guards，保持 material、persistence、cache request、geometry
  planning、geometry execution、legacy/outer qualification 和 observable 层的隔离；
- 所有新对象保持 diagnostic-only。

TODO 4 尚未完成的 qualification 项：

- SPM 与 d-wave 的真实 microscopic representative points；
- exact n=0 与至少一个正 Matsubara 频率；
- 轴向和斜向 q、零角和非零相对角；
- 完整匹配 material、primitive、phase、working N、primary shift 与 reduction 后的
  真实旧/新直接比较记录；
- 使用这些真实点值的小型 fixed-outer-Q replay 记录。

当前 d-wave representative responses 的 hard physical closure 已通过，但 provisional
response-space N/shift 门槛仍未闭合。下一步先审查 observable-impact calibration，
再决定哪些 point 值得继续扩大 N、是否需要修改积分策略，以及怎样建立
observable-specific error budget。不能因为 local logdet 对 response spread 不敏感，
就把 unresolved response 自动提升为 certified cache artifact。

这些只允许是窄范围 qualification，不允许扩展成新的大规模扫描。任何 unresolved
response 必须先生成完整诊断证据，不能直接通过扩大 N、放宽容差或替换代表点绕过。

后续仍未建立：

- TODO 5 的频率压缩与独立 holdout 误差合同；
- TODO 7 的 q/角度 surrogate response library；
- TODO 10 的完整 outer-q/Matsubara observable-level error budget 和生产准入；
- 真实 LNO327 四轨道模型的全栈资格。

## 几何与缓存边界

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

TODO 4 的 geometry plan 可以包含距离、板角和 `q_lab`，因为它描述的是几何任务；
这些字段不得反向污染材料响应身份。geometry executor 只接受 strict read-only
cache。缺失任一 exact response 时必须报告全部 miss 并终止，不允许 microscopic
fallback、cache write 或旧路线 fallback。

observable-impact calibration 是一个独立的 validation-only 旁路。它只能读取
unresolved diagnostic JSON 中已经持久化的 response values，并在独立输出中计算
reflection/logdet spread；它不允许向 geometry executor 或 certified cache 注入这些
response，也不能把 calibration 结果解释为 production admission。

旧的进程内 `CrystalResponseCache` 与历史 point cache 没有被升级、迁移或
重命名为新的 certified response cache。

## 旧路线与 qualification 的地位

`fixed_transverse_point_engine.py` 与
`fixed_transverse_point_certification.py` 仍保存旧的 geometry-specific
`two_plate_logdet` sweet-spot 路线，用于历史回归和归档结果解释。它们不是新的
可复用材料响应架构，也不能把旧 point cache 自动提升为 response cache。

旧路线只允许从 `material_geometry_legacy_replay.py` 和
`material_geometry_qualification.py` 的隔离边界进入显式比较。比较前必须验证
material state、response policy、primitive contract、phase policy、working N、
primary shift、canonical reduction、exact q、频率和角度全部匹配。

`material_geometry_outer_qualification.py` 只接收已经匹配的 old/new logdet arrays，
并用同一个 fixed outer grid 做有限 partial-sum replay。它不触发 microscopic 或
geometry 计算，也不构成 tail 或 observable-level error qualification。

新 geometry batch 或 strict cache-only 路线失败时，不允许自动退回旧链并把结果
标成等价。

## 接下来的顺序

```text
TODO 4 unresolved response evidence review
→ local observable-impact calibration from persisted N/shift histories
→ choose economical response/integration policy from measured propagation error
→ representative real old/new point qualification
→ real reduced fixed-outer replay
→ TODO 4 review/merge
→ TODO 5 finite-temperature frequency compression/reference-subtraction pilot
→ high-frequency convergence of Delta F / torque / pressure
→ observable-specific production qualification
```
