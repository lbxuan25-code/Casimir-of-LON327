# Casimir Local-Response 冒烟测试输出

本目录保存下列工程链路的冒烟测试输出：

$$
\mathrm{LocalSheetResponse}
\rightarrow \sigma_{\alpha\beta}
\rightarrow r
\rightarrow \mathcal{E}_{\mathrm{integrand}}
\rightarrow \tau_{\mathrm{integrand}} .
$$

这些输出不是正式 Casimir 能量或力矩计算。它们会先把 model response 转成
SI sheet conductivity，再接入 reflection matrix；但仍使用 local $q=0$ 响应矩阵，
跳过 $n=0$ 的正式物理处理，也尚未解决真实非局域 $q_{\parallel}$ 响应。
