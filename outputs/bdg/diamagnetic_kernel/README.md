# BdG diamagnetic kernel 输出

- `data/`：BdG diamagnetic kernel $K_{\mathrm{dia}}$ 的 `.npz` 诊断数据。
- `figures/`：$K_{\mathrm{dia}}$ 的对称性诊断图。

这些输出只用于 diamagnetic-kernel 诊断；它们不是
$K_{\mathrm{total}}$，也不是完整超导电导。
在当前 response contract 中它作为 mass/contact term，并通过
$K_{\mathrm{total}}=K_{\mathrm{dia}}-K_{\mathrm{para}}$ 进入 total kernel。
