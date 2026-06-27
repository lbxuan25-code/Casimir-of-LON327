# BdG finite-q 验证计划

本文档记录当前 superconducting BdG finite-q Ward closure blocker 的数值验证顺序。目标是定位定义、表示和 Goldstone 归一化问题，而不是通过拟合或修补响应来让 Ward 残差变小。

core finite-q 计算位于 `src/lno327/finite_q_engine.py`。可复用的基础诊断 helper 保留在 `src/lno327/finite_q_diagnostics.py`。项目级 validation workflow 位于 `validation/scripts/bdg_finite_q/`，轻量 summary 与复现命令位于 `validation/outputs/bdg_finite_q/`。

## 验证顺序

1. **q=0 响应定义对齐**  
   先比较 finite-q BdG engine 在 `q=[0, 0]` 时的 bubble、direct、total、phase Schur 和 amplitude-phase Schur 对象，与既有 local BdG / normal response 对象之间的约定差异。q=0 对齐是前置检查，不是最终目标。

2. **onsite_s finite-q Ward 控制检查**  
   用最简单的配对作为控制组，确认有限 q Ward 残差的数量级、Schur 修正方向和 collective kernel 条件数。

3. **spm finite-q Ward 控制检查**  
   检查当前 separable fixed-form-factor 模型下，`spm` 是否与 `onsite_s` 呈现一致或可解释的残差趋势。

4. **dwave 静态 pairing 重构**  
   在解释 `dwave` finite-q Ward 失败前，先确认代码中的 `dwave` pairing matrix 能由现有 bond/orbital 表示重构。若重构通过，不能把 finite-q 失败直接归因于静态 `dwave` ansatz。

5. **dwave endpoint-gauge tangent 检查**  
   检查 finite-q endpoint-gauge Goldstone tangent 在 q=0 是否回到全局相位 tangent，并确认 separable ansatz 使用的 tangent 与重构的 `dwave` 表示一致。

6. **Goldstone counterterm 与 eta2 归一化检查**  
   对 `onsite_s`、`spm`、`dwave` 检查 `K_eta2_eta2(q=0, omega=0)` 加 counterterm 后是否满足 Goldstone 条件，并确认元数据中的归一化约定是 `eta2 = delta0 * theta`。

7. **完整 finite-q Ward 残差扫描**  
   扫描小 q 和方向，分别报告 `bare_total`、`minus_schur`、`amplitude_phase_schur` 的左右 Ward 残差、残差比例、q scaling 估计、collective kernel 条件数和 inverse method。

8. **全部通过后再重新考虑 finite-q Casimir**  
   只有完成以上定义对齐和 Ward closure 诊断后，才可以重新讨论 formal finite-q Casimir。当前任务不运行 formal finite-q Casimir，也不做 torque 结论。

## 当前边界

- 最终 blocker 仍然是 superconducting BdG finite-q Ward closure。
- `dwave` failure 在完成重构和 tangent 检查前，不能解释为静态 `dwave` ansatz 错误。
- separable fixed-form-factor 仍是当前 production model。
- 可能出现的 massive internal pairing-shape modes 只能作为诊断线索，不是新的 Goldstone modes，也不是 production modes。
- finite-q 诊断输出始终是 `valid_for_casimir_input=False`，不是 Casimir-ready 输入。
- Ward validation 只检查残差，不修补 response；禁止 response-level 拟合、残差投影或修复。
- `validation/scripts/bdg_finite_q/` 下的 workflow 是 validation 层，不是稳定 public API；raw arrays、大型 CSV、expanded logs 和 cache tensors 不应提交到 `validation/outputs/bdg_finite_q/`。
