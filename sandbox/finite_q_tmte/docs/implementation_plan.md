# 实现计划

## 模块布局

- `tmte/theory`：纯数学和约定。包含 Matsubara `n -> xi_eV` 频率约定、q 方向、目标基、顶点线性组合、contact 投影、Schur 线性代数。
- `tmte/adapters`：唯一允许接触现有模型和 BdG 有限 q 后端的层。这里把现有 primitive 顶点、bubble、collective block 封装成目标基 blocks。
- `tmte/pipeline`：扫描编排、shifted mesh 平均、诊断和 JSON schema。
- `tmte/io`：complex JSON 读写。
- `scripts`：薄 CLI wrapper，只解析参数并调用 pipeline。
- `tests`：理论层和 pipeline 轻量测试，不运行昂贵物理扫描。

## 依赖方向

```text
scripts
  -> pipeline
      -> theory + adapters + io
          -> existing src code only through adapters
```

禁止方向：

- `src` 不导入 `sandbox`。
- `tmte/theory` 不导入 `tmte/adapters` 或 `tmte/pipeline`。
- `tmte/adapters` 不导入 `tmte/pipeline`。
- CLI 脚本不写物理公式。

## 实现顺序

1. 写定约定与推导文档。
2. 实现 `theory` 中的 Matsubara 频率、q 基、目标顶点、contact 和 Schur。
3. 实现 adapter：模型选择、primitive 顶点、target-basis bubble、collective block。
4. 实现 block builder、shifted-average、diagnostics、schema 与 writer。
5. 增加薄 CLI。
6. 加轻量测试与 compile/pytest 检查。
