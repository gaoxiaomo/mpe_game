# $n$追一通信感知组级追捕模型的数学推导

这是一份更简洁的 MathJax 版推导，只保留数学主线：建模、Hamiltonian、控制律、稳定性与通信中断后的重新收敛。

对应 Overleaf 版本：

- [TEAM_COMM_DERIVATION_OVERLEAF_CN.tex](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/TEAM_COMM_DERIVATION_OVERLEAF_CN.tex)

---

## 1. 建模思路

考虑 $n$ 架追逐者共同追捕一架逃避者的场景。若这 $n$ 架追逐者的目标相同，则更自然的建模方式不是把它们分别视为彼此独立的最优控制器，而是将其视为一个追捕小组，用一个组级值函数统一刻画协同追捕行为。

因此，我们将问题写成：

1. $n$ 个追逐者联合最小化同一个组级性能指标。
2. 逃避者在理论上最大化该性能指标。
3. 通信正常时使用完整组状态。
4. 通信中断时每个追逐者仅使用按通信矩阵掩码后的组观测 $\bar Z_j=\Gamma_j Z$。
5. 被屏蔽块直接零填充，并通过可见性掩码同步关闭对应特征与交叉项；通信恢复后立即重新同步到真实组状态。

---

## 2. 单体动力学与相对误差

每个智能体采用六维状态与三维控制输入：

$$
 x=[x,y,h,v_x,v_y,v_h]^T\in\mathbb{R}^6,
 \qquad
 u=[u_1,u_2,u_3]^T\in\mathbb{R}^3.
$$

其动力学写为仿射非线性形式：

$$
\dot x = f(x)+g(x)u.
$$

对第 $j$ 个追逐者，有

$$
\dot x_j^p = f_j(x_j^p)+g_j(x_j^p)u_j^p,
\qquad j=1,\dots,n,
$$

对逃避者有

$$
\dot x^e = f_e(x^e)+g_e(x^e)u^e.
$$

对每个追逐者定义相对误差：

$$
\tilde x_j = x_j^p - x^e + r_j,
\qquad j=1,\dots,n,
$$

其中 $r_j\in\mathbb{R}^6$ 为期望位移。

其导数为

$$
\dot{\tilde x}_j
=
 f_j(x_j^p)-f_e(x^e)
 +g_j(x_j^p)u_j^p
 -g_e(x^e)u^e
 +\dot r_j.
$$

若 $r_j$ 为常量，则 $\dot r_j=0$。

---

## 3. 组级状态与组级性能指标

将 $n$ 个误差块堆叠成组状态：

$$
Z = \operatorname{col}(\tilde x_1,\tilde x_2,\dots,\tilde x_n)\in\mathbb{R}^{6n}.
$$

将 $n$ 个追逐者的控制堆叠为

$$
U^p = \operatorname{col}(u_1^p,u_2^p,\dots,u_n^p)\in\mathbb{R}^{3n}.
$$

于是组状态动力学可写成

$$
\dot Z = F(Z)+G(Z)U^p+E(Z)u^e.
$$

定义组级运行代价

$$
\ell(Z,U^p,u^e)
=
\sum_{j=1}^n P_j(\tilde x_j)
+
\sum_{j=1}^n U_j(u_j^p)
-W(u^e).
$$

从而组级性能指标为

$$
J_{\mathcal S}(Z(0),U^p,u^e)
=
\int_0^{\infty} \ell(Z,U^p,u^e)\,dt.
$$

这里，状态代价取为

$$
P_j(\tilde x_j)
=
\left(\frac{\tilde x_j}{s}\right)^T Q \left(\frac{\tilde x_j}{s}\right),
$$

其中 $s\in\mathbb{R}^6$ 为状态归一化尺度，$Q\succ 0$。

为保持与原始饱和控制结构一致，输入代价采用非二次形式。对追逐者，记

$$
U_j(u_j^p)
=
\sum_{\ell=1}^{3}
2\bar u_p r_{1,\ell}
\left[
 u_{j,\ell}^p\operatorname{atanh}\!\left(\frac{u_{j,\ell}^p}{\bar u_p}\right)
 +
 \frac{\bar u_p}{2}\ln\!\left(1-\left(\frac{u_{j,\ell}^p}{\bar u_p}\right)^2\right)
\right],
$$

对逃避者，记

$$
W(u^e)
=
\sum_{\ell=1}^{3}
2\bar u_e r_{2,\ell}
\left[
 u_{\ell}^e\operatorname{atanh}\!\left(\frac{u_{\ell}^e}{\bar u_e}\right)
 +
 \frac{\bar u_e}{2}\ln\!\left(1-\left(\frac{u_{\ell}^e}{\bar u_e}\right)^2\right)
\right].
$$

---

## 4. 组级 Hamiltonian 与 HJI 方程

定义组级值函数

$$
V(Z)=\min_{U^p}\max_{u^e} J_{\mathcal S}(Z,U^p,u^e).
$$

则 Hamiltonian 为

$$
H(Z,U^p,u^e,\nabla V)
=
\ell(Z,U^p,u^e)
+
\nabla V(Z)^T\big(F(Z)+G(Z)U^p+E(Z)u^e\big).
$$

对应 HJI 方程为

$$
0 = \min_{U^p}\max_{u^e} H(Z,U^p,u^e,\nabla V).
$$

这是整个组级追逃问题的核心数学对象。

---

## 5. 最优控制律推导

将组级梯度按块写成

$$
\nabla V(Z)
=
\operatorname{col}\left(
\frac{\partial V}{\partial \tilde x_1},
\frac{\partial V}{\partial \tilde x_2},
\dots,
\frac{\partial V}{\partial \tilde x_n}
\right).
$$

### 5.1 追逐者控制律

对第 $j$ 个追逐者，令

$$
\frac{\partial H}{\partial u_j^p}=0.
$$

由于上述非二次输入代价满足

$$
\frac{\partial U_j}{\partial u_j^p}
=
2\bar u_p R_1\,\operatorname{atanh}\!\left(\frac{u_j^p}{\bar u_p}\right),
$$

因此驻值条件化为

$$
2\bar u_p R_1\,\operatorname{atanh}\!\left(\frac{u_j^{p*}}{\bar u_p}\right)
+
g_j(x_j^p)^T\frac{\partial V}{\partial \tilde x_j}=0.
$$

两边同乘 $R_1^{-1}/(2\bar u_p)$，得

$$
\operatorname{atanh}\!\left(\frac{u_j^{p*}}{\bar u_p}\right)
=
-\frac{1}{2\bar u_p}R_1^{-1}g_j(x_j^p)^T\frac{\partial V}{\partial \tilde x_j}.
$$

再取双曲正切，可得

$$
u_j^{p*}
=
-\bar u_p
\tanh\!\left(
\frac{1}{2\bar u_p}
R_1^{-1}
g_j(x_j^p)^T
\frac{\partial V}{\partial \tilde x_j}
\right),
\qquad j=1,\dots,n.
$$

这说明每个追逐者的控制仍保持原始论文中的饱和 $\tanh$ 结构，只是梯度项不再来自单个 critic，而是来自同一个组级值函数的分块梯度。

### 5.2 逃避者控制律

对逃避者做极大化，令

$$
\frac{\partial H}{\partial u^e}=0.
$$

同理可得

$$
u^{e*}
=
-\bar u_e
\tanh\!\left(
\frac{1}{2\bar u_e}
R_2^{-1}
g_e(x^e)^T
\sum_{j=1}^{3}\frac{\partial V}{\partial \tilde x_j}
\right).
$$

因此，从理论上看，逃避者面对的是整个追捕小组的综合梯度压力。

---

## 6. 值函数近似

用线性参数化形式逼近组级值函数：

$$
V(Z)\approx \hat V(Z)=\hat W^T\phi(Z),
$$

其中 $\hat W$ 为待学习权重，$\phi(Z)$ 为组级特征向量。

于是梯度近似为

$$
\nabla \hat V(Z)
=
\left(\frac{\partial \phi(Z)}{\partial Z}\right)^T\hat W.
$$

第 $j$ 个追逐者对应的分块梯度为

$$
\frac{\partial \hat V}{\partial \tilde x_j}
=
\left(\frac{\partial \phi(Z)}{\partial \tilde x_j}\right)^T\hat W.
$$

因此，近似控制律写为

$$
u_j^p
=
-\bar u_p
\tanh\!\left(
\frac{1}{2\bar u_p}
R_1^{-1}
g_j(x_j^p)^T
\left(\frac{\partial \phi(Z)}{\partial \tilde x_j}\right)^T\hat W
\right),
$$

以及

$$
u^e
=
-\bar u_e
\tanh\!\left(
\frac{1}{2\bar u_e}
R_2^{-1}
g_e(x^e)^T
\sum_{j=1}^{3}\left(\frac{\partial \phi(Z)}{\partial \tilde x_j}\right)^T\hat W
\right).
$$

执行阶段若通信受限，则第 $j$ 个追逐者实际使用

$$
\bar Z_j=\Gamma_j Z,
\qquad
\Gamma_j=\operatorname{diag}(\gamma_{1j}I_6,\dots,\gamma_{nj}I_6),
\qquad
m_j=[\gamma_{1j},\dots,\gamma_{nj}]^T.
$$

当前实现对应的是掩码特征映射 $\phi(\bar Z_j;m_j)$：局部特征按 $\gamma_{kj}$ 缩放，交叉特征按 $\gamma_{aj}\gamma_{bj}$ 缩放，因此隐藏块在当前时刻被直接零填充，而不是引入历史保持块。

---

## 7. 无通信中断时的稳定性

若采用理想值函数 $V^*(Z)$，则其可作为 Lyapunov 候选函数。沿最优闭环轨迹有

$$
\dot V^*(Z)
=
\nabla V^*(Z)^T\dot Z
=
-\ell\big(Z,U^{p*},u^{e*}\big).
$$

若运行代价对组状态是正定的，即

$$
\ell\big(Z,U^{p*},u^{e*}\big)\ge \alpha\|Z\|^2,
\qquad \alpha>0,
$$

则有

$$
\dot V^*(Z)\le -\alpha\|Z\|^2\le 0.
$$

这表明理想闭环下系统稳定。

**定理 1**

若理想值函数 $V^*(Z)$ 正定且径向无界，并满足上述 HJI 方程，则无通信中断时的理想闭环系统渐近稳定。

**证明**

由 $V^*(Z)$ 的正定性与径向无界性可知，它可以作为 Lyapunov 函数。又由 HJI 方程在最优控制下成立，有

$$
\dot V^*(Z)=-\ell\big(Z,U^{p*},u^{e*}\big)\le 0.
$$

若 $\ell$ 对误差状态正定，则除平衡点外有 $\dot V^*(Z)<0$，故由 Lyapunov 渐近稳定定理可知系统渐近稳定。

---

## 8. 近似值函数下的一致最终有界性

实际实现中使用的是近似值函数 $\hat V(Z)$，而不是理想值函数 $V^*(Z)$。设近似误差满足

$$
\nabla \hat V(Z)=\nabla V^*(Z)+\varepsilon_V(Z),
\qquad \|\varepsilon_V(Z)\|\le \bar\varepsilon_V.
$$

则沿近似闭环轨迹有

$$
\dot{\hat V}(Z)
\le
-\alpha_1\|Z\|^2 + \alpha_2\bar\varepsilon_V,
$$

其中 $\alpha_1,\alpha_2>0$。

因此，当 $\|Z\|$ 足够大时，负项占主导，系统轨迹最终进入一个由近似误差决定的有界邻域。

**定理 2**

在值函数近似误差有界、控制输入有界、动力学与特征映射局部 Lipschitz 的条件下，近似闭环系统是一致最终有界的。

**证明思路**

1. 将 $\hat V$ 视作近似 Lyapunov 函数。
2. 理想部分给出负定衰减，近似误差只引入有界扰动项。
3. 因而当 $\|Z\|$ 超过某个阈值后，有 $\dot{\hat V}(Z)<0$。
4. 系统状态最终被吸引到一个由 $\bar\varepsilon_V$ 决定的有界集合中。

---

## 9. 通信中断与恢复下的重新收敛

通信正常时，每个追逐者都可获得完整组状态 $Z$。通信中断时，第 $j$ 个追逐者只能使用掩码组观测 $\bar Z_j=\Gamma_j Z$，并定义观测误差

$$
e_j^Z = \bar Z_j - Z = -(I-\Gamma_j)Z.
$$

此时，第 $j$ 个追逐者的实际执行控制可写为

$$
u_j^{p,\mathrm{exec}}
=
-\bar u_p
\tanh\!\left(
\frac{1}{2\bar u_p}
R_1^{-1}
g_j(x_j^p)^T
\left(\frac{\partial \phi(\bar Z_j;m_j)}{\partial \tilde x_j}\right)^T\hat W
\right).
$$

由 $\tanh(\cdot)$ 的 Lipschitz 性以及特征梯度的有界性，存在常数 $L_u>0$，使得

$$
\|u_j^{p,\mathrm{exec}}-u_j^{p*}\|
\le
L_u\|e_j^Z\|.
$$

因此，通信中断的作用可以视为闭环系统中的一个由掩码观测引起的有界扰动项。只要中断持续时间有限、恢复后 $\Gamma_j\to I$，则该扰动也是有限时长的。

**定理 3**

若无通信中断时近似闭环系统为一致最终有界，且通信中断时间有限、恢复后 $\Gamma_j\to I$ 从而 $e_j^Z\to 0$，则通信中断/恢复后的闭环系统仍保持有界，并在恢复后重新进入原一致最终有界邻域。

**证明思路**

中断期间，控制误差满足

$$
\|u_j^{p,\mathrm{exec}}-u_j^{p*}\|\le L_u\|e_j^Z\|,
$$

因此闭环系统等价于理想近似闭环系统叠加一个有界扰动。由于中断时长有限，扰动总能量有限；通信恢复后，$e_j^Z\to 0$，扰动项消失。由一致最终有界系统对有界扰动的鲁棒性可知，系统状态始终保持有界，并在恢复后重新回到原有界邻域。

---

## 10. 可行性结论

由上述推导可知，这个 $n$ 追一通信感知组级追捕模型在数学上是自洽的：

1. 组级性能指标、Hamiltonian 与 HJI 方程定义明确。
2. 最优控制律仍保持与原始论文一致的饱和 $\tanh$ 结构。
3. 使用组级值函数后，所有追逐者的控制天然耦合到同一个组级梯度。
4. 通信中断可等价视为局部组状态估计误差导致的有界扰动。
5. 在理想值函数下可得到渐近稳定，在近似值函数下可得到一致最终有界，在中断/恢复下可得到恢复后的重新收敛。

因此，这一模型足以作为“多追一场景下，追逐者之间通信是否有理论意义、通信中断后是否仍能恢复收敛”的数学验证框架。


补充说明：当前工程中的数值实验仍然使用 $n=3$ 的特例，但上述推导本身针对的是一般的 $n$ 追一场景。
