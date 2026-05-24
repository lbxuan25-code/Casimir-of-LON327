# 基础说明

本仓库当前只实现底层代数结构和物理流程接口，不执行正式物理数值模拟，
也不声明任何最终数值结论。当前工作重心是先研究 `s_pm` 与 `d_wave`
在 gap structure 和 conductivity symmetry 上的区别；Casimir torque 是后续应用层。

## Normal-State 模型

normal-state Hamiltonian 使用四轨道基
`(dz1, dx1, dz2, dx2)`：

`H(k) = [[H_parallel, H_perp], [H_perp, H_parallel]] - mu I`.

代码中实现的系数是项目正式采用的 `Tz_k`、`Tx_k`、`Tz_perp,k`、
`Tx_perp,k`、`V_k` 与 `V'_k`。化学势作为 normal-state 参数保存，
取值为 `mu = 0.05 eV`。

`s_pm` 配对采用 `(dz1, dx1, dz2, dx2)` 基下的层间 dz2 结构：

`Delta_s_pm = delta0_eV * [[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]`.

它表示 bilayer bonding/antibonding sign-changing s_pm pairing。`d_wave`
配对采用同层 dz2-dx2_y2 interorbital 结构：

`Delta_d = delta0_eV * (cos(kx) + cos(ky)) * [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]`.

其中动量因子是 A1g，结合 dx2_y2 轨道自身的 B1g 对称性后，总配对属于
d-wave/B1g 通道。两类配对矩阵均为偶宇称 spin-singlet 形式，满足
`Delta(k) = Delta^T(-k)`。

所有 Hamiltonian、配对、速度顶点和 Kubo 响应中的能量单位均为 eV。
Kubo 使用的速度算符是 `dH/dk_alpha`，其单位也按 eV 处理，因为 `kx, ky`
是无量纲晶格动量。需要 SI 输出时，Kubo 电导会把无量纲能带响应乘以
`e^2/hbar`。玻色 Matsubara 能量写作 `hbar xi_n = 2 pi n kBT`，单位为 eV。

## Dai and Jiang

卡西米尔相关工具实现以下流程骨架：

1. 虚频下的电导张量；
2. 按板间相对角度/面内角度旋转张量；
3. 构造反射矩阵；
4. 构造 Lifshitz 能量 integrand；
5. 由 `-partial_theta E` 构造力矩 integrand。

Kubo 电导被明确拆分为虚频轴 `sigma(i xi)` 与实频轴 `sigma(omega)` 两个函数，
二者都使用 normal-state 速度顶点。BZ 积分规范为 `sum_k w_k` 且
`sum w_k = 1`，对应在 `[-pi, pi)^2` 上的 `int_BZ d2k/(2 pi)^2`。

## 当前执行边界

当前阶段优先维护：

1. normal-state Hamiltonian 与 pairing ansatz；
2. BdG 谱、gap projection、near-node 诊断；
3. normal-state conductivity symmetry 基线；
4. BdG paramagnetic kernel 基础层。

完整 superconducting conductivity 需要后续加入 diamagnetic term 后再命名和使用。
Casimir torque 暂时只保留接口骨架，不用来给出物理判断。
