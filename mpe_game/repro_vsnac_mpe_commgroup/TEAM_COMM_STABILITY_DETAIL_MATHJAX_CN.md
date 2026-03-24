# $n$追一通信感知组级追捕模型的稳定性与一致性详细推导

这份文档只讨论两件事：

1. 组级控制律在数学上是否自洽。
2. 通信正常、通信中断、通信恢复三种情况下，闭环系统的稳定性与一致性如何理解。

这里不展开工程训练细节，不讨论 replay、阻尼、回溯更新等实现问题，只保留数理推导主线。

对应的 Overleaf 版本：

- [TEAM_COMM_STABILITY_DETAIL_OVERLEAF_CN.tex](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/TEAM_COMM_STABILITY_DETAIL_OVERLEAF_CN.tex)

---

## 1. 组级追捕模型

### 1.1 单体动力学

每个智能体的状态为

$$
x=[x,y,h,v_x,v_y,v_h]^T\in\mathbb{R}^6,
$$

控制输入为

$$
u=[u_1,u_2,u_3]^T\in\mathbb{R}^3,
$$

动力学写为

$$
\dot x = f(x)+g(x)u.
$$

对第 $j$ 个追逐者：

$$
\dot x_j^p = f_j(x_j^p)+g_j(x_j^p)u_j^p,\qquad j=1,\dots,n.
$$

对逃避者：

$$
\dot x^e = f_e(x^e)+g_e(x^e)u^e.
$$

### 1.2 相对误差

定义第 $j$ 个追逐者的相对误差

$$
\tilde x_j = x_j^p - x^e + r_j,
$$

其中 $r_j\in\mathbb{R}^6$ 为期望位移。

若 $r_j$ 为常量，则

$$
\dot{\tilde x}_j
=
f_j(x_j^p)-f_e(x^e)
+g_j(x_j^p)u_j^p
-g_e(x^e)u^e.
$$

### 1.3 组级状态

将 $n$ 个误差块堆叠为组级状态

$$
Z=\operatorname{col}(\tilde x_1,\tilde x_2,\dots,\tilde x_n)\in\mathbb{R}^{6n}.
$$

将 $n$ 个追逐者控制堆叠为

$$
U^p=\operatorname{col}(u_1^p,u_2^p,\dots,u_n^p)\in\mathbb{R}^{3n}.
$$

则组状态动力学可写成

$$
\dot Z = F(Z)+G(Z)U^p+E(Z)u^e.
$$

---

## 2. 组级性能指标与 Hamiltonian

定义运行代价

$$
\ell(Z,U^p,u^e)
=
\sum_{j=1}^{n}P_j(\tilde x_j)
+
\sum_{j=1}^{n}U_j(u_j^p)
-
W(u^e).
$$

定义组级性能指标

$$
J_{\mathcal S}(Z(0),U^p,u^e)
=
\int_0^\infty \ell(Z,U^p,u^e)\,dt.
$$

其中状态代价取为

$$
P_j(\tilde x_j)
=
\left(\frac{\tilde x_j}{s}\right)^T Q \left(\frac{\tilde x_j}{s}\right),
\qquad Q\succ 0.
$$

定义组级值函数

$$
V(Z)=\min_{U^p}\max_{u^e}J_{\mathcal S}(Z,U^p,u^e).
$$

则 Hamiltonian 为

$$
H(Z,U^p,u^e,\nabla V)
=
\ell(Z,U^p,u^e)+\nabla V(Z)^T\big(F(Z)+G(Z)U^p+E(Z)u^e\big).
$$

对应 HJI 方程为

$$
0=\min_{U^p}\max_{u^e}H(Z,U^p,u^e,\nabla V).
$$

---

## 3. 最优控制律一步一步推导

### 3.1 梯度分块

将组级梯度写成分块形式

$$
\nabla V(Z)
=
\operatorname{col}\!\left(
\frac{\partial V}{\partial \tilde x_1},
\frac{\partial V}{\partial \tilde x_2},
\dots,
\frac{\partial V}{\partial \tilde x_n}
\right).
$$

### 3.2 追逐者控制律

因为 $u_j^p$ 只出现在 $U_j(u_j^p)$ 和第 $j$ 个误差块动力学中，所以对 Hamiltonian 关于 $u_j^p$ 求偏导，有

$$
\frac{\partial H}{\partial u_j^p}
=
\frac{\partial U_j}{\partial u_j^p}
+
g_j(x_j^p)^T\frac{\partial V}{\partial \tilde x_j}.
$$

对一维输入分量，记

$$
\varphi(u)
=
2\bar u\,r
\left[
u\operatorname{atanh}\!\left(\frac{u}{\bar u}\right)
+
\frac{\bar u}{2}\ln\!\left(1-\left(\frac{u}{\bar u}\right)^2\right)
\right].
$$

则有

$$
\frac{d\varphi}{du}
=
2\bar u\,r\,\operatorname{atanh}\!\left(\frac{u}{\bar u}\right).
$$

因此对向量形式，

$$
\frac{\partial U_j}{\partial u_j^p}
=
2\bar u_p R_1\,\operatorname{atanh}\!\left(\frac{u_j^p}{\bar u_p}\right).
$$

驻值条件 $\frac{\partial H}{\partial u_j^p}=0$ 给出

$$
2\bar u_p R_1\,\operatorname{atanh}\!\left(\frac{u_j^{p*}}{\bar u_p}\right)
+
g_j(x_j^p)^T\frac{\partial V}{\partial \tilde x_j}
=0.
$$

整理为

$$
\operatorname{atanh}\!\left(\frac{u_j^{p*}}{\bar u_p}\right)
=
-\frac{1}{2\bar u_p}
R_1^{-1}
g_j(x_j^p)^T
\frac{\partial V}{\partial \tilde x_j}.
$$

两边取双曲正切，得到

$$
u_j^{p*}
=
-\bar u_p
\tanh\!\left(
\frac{1}{2\bar u_p}
R_1^{-1}
g_j(x_j^p)^T
\frac{\partial V}{\partial \tilde x_j}
\right).
$$

### 3.3 逃避者控制律

对逃避者做极大化，有

$$
\frac{\partial H}{\partial u^e}
=
-\frac{\partial W}{\partial u^e}
+
E(Z)^T\nabla V(Z).
$$

由于

$$
E(Z)=-\operatorname{col}(g_e(x^e),g_e(x^e),\dots,g_e(x^e)),
$$

故

$$
E(Z)^T\nabla V(Z)
=
-g_e(x^e)^T
\sum_{j=1}^{n}\frac{\partial V}{\partial \tilde x_j}.
$$

再利用

$$
\frac{\partial W}{\partial u^e}
=
2\bar u_e R_2\,\operatorname{atanh}\!\left(\frac{u^e}{\bar u_e}\right),
$$

驻值条件得到

$$
2\bar u_e R_2\,\operatorname{atanh}\!\left(\frac{u^{e*}}{\bar u_e}\right)
+
g_e(x^e)^T\sum_{j=1}^{n}\frac{\partial V}{\partial \tilde x_j}
=0.
$$

从而

$$
u^{e*}
=
-\bar u_e
\tanh\!\left(
\frac{1}{2\bar u_e}
R_2^{-1}
g_e(x^e)^T
\sum_{j=1}^{n}\frac{\partial V}{\partial \tilde x_j}
\right).
$$

---

## 4. 一致性推导

### 4.1 定义

定义真实组状态

$$
Z=\operatorname{col}(\tilde x_1,\tilde x_2,\dots,\tilde x_n).
$$

定义第 $j$ 个追逐者的掩码组观测

$$
\bar Z_j=\Gamma_j Z,
\qquad
\Gamma_j=\operatorname{diag}(\gamma_{1j}I_6,\gamma_{2j}I_6,\dots,\gamma_{nj}I_6).
$$

定义第 $j$ 个追逐者的估计误差

$$
e_j^Z=\bar Z_j-Z.
$$

### 4.2 全通信下的一步一致性

设通信矩阵为 $\Gamma(t)=[\gamma_{ab}(t)]\in\{0,1\}^{n\times n}$，其中

$$
\gamma_{ab}(t)=1
$$

表示发送者 $a$ 的状态块在时刻 $t$ 能被接收者 $b$ 获得。

**命题 1**

若在某时刻 $t$ 有

$$
\gamma_{bj}(t)=1,\qquad \forall b,j\in\{1,2,\dots,n\},
$$

则更新后

$$
\bar Z_j=Z,\qquad \forall j.
$$

**证明**

当对所有 $b,j$ 都有 $\gamma_{bj}(t)=1$ 时，有 $\Gamma_j=I$，从而

$$
\bar Z_j=\Gamma_j Z=IZ=Z.
$$

故命题成立。

### 4.3 全通信下的控制一致性

**命题 2**

若 $\bar Z_j=Z$ 且 $m_j=\mathbf 1_n$，则第 $j$ 个追逐者的分布式执行控制与集中式组控制完全一致。

**证明**

第 $j$ 个追逐者执行控制为

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

当 $\bar Z_j=Z$ 且 $m_j=\mathbf 1_n$ 时，

$$
\left(\frac{\partial \phi(\bar Z_j;m_j)}{\partial \tilde x_j}\right)^T\hat W
=
\left(\frac{\partial \phi(Z)}{\partial \tilde x_j}\right)^T\hat W,
$$

故

$$
u_j^{p,\mathrm{exec}}=u_j^{p,\mathrm{cen}}.
$$

### 4.4 恢复通信后的重新一致性

**命题 3**

若在断联之后的某个恢复时刻 $t_r$ 起重新满足

$$
\gamma_{bj}(t_r)=1,\qquad \forall b,j,
$$

则在恢复后的第一次估计更新后有

$$
e_j^Z(t_r^+)=0,\qquad \forall j.
$$

**证明**

由命题 1，恢复后有 $\Gamma_j(t_r)=I$，于是 $\bar Z_j(t_r^+)=Z$。故

$$
e_j^Z(t_r^+)=\bar Z_j(t_r^+)-Z=0.
$$

---

## 5. 稳定性详细推导

### 5.1 第一步：运行代价的下界

记

$$
S=\operatorname{diag}(s_1,\dots,s_6),
\qquad
\bar Q=\operatorname{diag}(\underbrace{S^{-T}QS^{-1},S^{-T}QS^{-1},\dots,S^{-T}QS^{-1}}_{n\text{ 个块}}).
$$

则有

$$
\sum_{j=1}^{n}P_j(\tilde x_j)=Z^T\bar Q Z.
$$

由于 $Q\succ 0$，故 $\bar Q\succ 0$。记

$$
\lambda_Q=\lambda_{\min}(\bar Q)>0,
$$

则

$$
\sum_{j=1}^{n}P_j(\tilde x_j)\ge \lambda_Q\|Z\|^2.
$$

另一方面，非二次输入代价满足

$$
U_j(u_j^p)\ge 0,\qquad W(u^e)\ge 0.
$$

由于逃避者输入有界，即

$$
\|u^e\|_{\infty}\le \bar u_e,
$$

故存在常数 $\bar W_e>0$ 使得

$$
W(u^e)\le \bar W_e.
$$

于是运行代价满足

$$
\ell(Z,U^p,u^e)
\ge
\lambda_Q\|Z\|^2-\bar W_e.
$$

### 5.2 第二步：理想值函数下的导数估计

若采用理想值函数 $V^*(Z)$，沿最优闭环轨迹有

$$
\dot V^*(Z)
=
\nabla V^*(Z)^T\dot Z.
$$

由 HJI 方程在鞍点处成立，有

$$
\dot V^*(Z)=-\ell(Z,U^{p*},u^{e*}).
$$

结合上一节的下界，得到

$$
\dot V^*(Z)
\le
-\lambda_Q\|Z\|^2+\bar W_e.
$$

这已经给出了一个标准的 UUB 型微分不等式。

### 5.3 第三步：为什么这里先得到 UUB，而不是直接得到渐近稳定

如果运行代价满足更强条件

$$
\ell(Z,U^{p*},u^{e*})\ge c\|Z\|^2,\qquad c>0,
$$

则可直接得到

$$
\dot V^*(Z)\le -c\|Z\|^2,
$$

从而推出渐近稳定。

但是在当前 $n$ 追一建模里，运行代价含有

$$
-W(u^e),
$$

而逃避者输入虽然有界，却不一定为零，因此全局上更自然的结论是

$$
\dot V^*(Z)\le -\lambda_Q\|Z\|^2+\bar W_e,
$$

也就是一致最终有界，而不是直接声称严格渐近稳定。

这一点是我们主动做出的理论收敛边界审查：对当前模型，更稳妥且合理的主结论是 UUB。

### 5.4 第四步：UUB 结论

**定理 1**

若 $V^*(Z)$ 正定、径向无界，且 HJI 方程在鞍点处成立，则理想组级闭环系统一致最终有界。

**证明**

由

$$
\dot V^*(Z)\le -\lambda_Q\|Z\|^2+\bar W_e
$$

可知，当

$$
\|Z\| > \sqrt{\frac{\bar W_e}{\lambda_Q}}
$$

时，有

$$
\dot V^*(Z)<0.
$$

因此所有轨迹最终都会进入球域

$$
\mathcal B_r=\left\{Z:\|Z\|\le \sqrt{\frac{\bar W_e}{\lambda_Q}}\right\}
$$

并保持在其中。故系统一致最终有界。

---

## 6. 近似值函数下的稳定性

### 6.1 近似误差进入闭环系统的方式

实际中使用近似值函数

$$
\hat V(Z)=\hat W^T\phi(Z),
$$

并假设其梯度误差有界：

$$
\left\|
\frac{\partial \hat V}{\partial \tilde x_j}
-
\frac{\partial V^*}{\partial \tilde x_j}
\right\|
\le \bar\varepsilon_j,\qquad j=1,\dots,n.
$$

定义

$$
\rho_j^*
=
\frac{1}{2\bar u_p}
R_1^{-1}g_j(x_j^p)^T\frac{\partial V^*}{\partial \tilde x_j},
$$

$$
\hat\rho_j
=
\frac{1}{2\bar u_p}
R_1^{-1}g_j(x_j^p)^T\frac{\partial \hat V}{\partial \tilde x_j}.
$$

由于 $\tanh(\cdot)$ 是 $1$-Lipschitz，故

$$
\|u_j^p-u_j^{p*}\|
=
\bar u_p\|\tanh(\hat\rho_j)-\tanh(\rho_j^*)\|
\le
\bar u_p\|\hat\rho_j-\rho_j^*\|.
$$

再利用范数不等式，

$$
\|u_j^p-u_j^{p*}\|
\le
\frac{1}{2}\|R_1^{-1}\|\|g_j(x_j^p)\|\,\bar\varepsilon_j.
$$

若在考虑区域内存在常数 $\bar g_p$ 使

$$
\|g_j(x_j^p)\|\le \bar g_p,
$$

则

$$
\|u_j^p-u_j^{p*}\|
\le
c_{p,j}\bar\varepsilon_j,
\qquad
c_{p,j}=\frac{1}{2}\|R_1^{-1}\|\bar g_p.
$$

同理，对逃避者有

$$
\|u^e-u^{e*}\|
\le
c_e\sum_{j=1}^{n}\bar\varepsilon_j.
$$

### 6.2 近似闭环导数估计

将近似控制与理想控制的差异视为等效扰动项 $d_V(Z)$，则闭环系统写为

$$
\dot Z = F_{\mathrm{ideal}}(Z)+d_V(Z),
$$

其中

$$
\|d_V(Z)\|\le \bar d_V.
$$

沿理想值函数 $V^*(Z)$ 求导，有

$$
\dot V^*(Z)
=
\nabla V^*(Z)^TF_{\mathrm{ideal}}(Z)
+
\nabla V^*(Z)^Td_V(Z).
$$

第一项由理想 HJI 给出

$$
\nabla V^*(Z)^TF_{\mathrm{ideal}}(Z)
\le
-\lambda_Q\|Z\|^2+\bar W_e.
$$

对第二项，若在所考察区域内

$$
\|\nabla V^*(Z)\|\le L_V\|Z\|,
$$

则

$$
\nabla V^*(Z)^Td_V(Z)
\le
L_V\bar d_V\|Z\|.
$$

利用 Young 不等式

$$
ab\le \frac{\eta}{2}a^2+\frac{1}{2\eta}b^2,
$$

取 $a=\|Z\|$、$b=L_V\bar d_V$，得

$$
L_V\bar d_V\|Z\|
\le
\frac{\eta}{2}\|Z\|^2+\frac{L_V^2\bar d_V^2}{2\eta}.
$$

因此

$$
\dot V^*(Z)
\le
-\left(\lambda_Q-\frac{\eta}{2}\right)\|Z\|^2
+
\bar W_e
+
\frac{L_V^2\bar d_V^2}{2\eta}.
$$

只要选择

$$
0<\eta<2\lambda_Q,
$$

则负二次项仍然保留，从而得到近似闭环系统的 UUB 结论。

**定理 2**

若梯度近似误差有界、$g_j(\cdot)$ 有界、$V^*(Z)$ 梯度满足局部线性增长条件，则近似 team critic 闭环系统一致最终有界。

---

## 7. 通信中断下的稳定性与恢复后的重新一致性

### 7.1 控制误差对估计误差的 Lipschitz 界

中断时，第 $j$ 个追逐者执行控制为

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

而全通信集中式近似控制为

$$
u_j^{p,\mathrm{cen}}
=
-\bar u_p
\tanh\!\left(
\frac{1}{2\bar u_p}
R_1^{-1}
g_j(x_j^p)^T
\left(\frac{\partial \phi(Z)}{\partial \tilde x_j}\right)^T\hat W
\right).
$$

若分块梯度映射

$$
Z\mapsto \left(\frac{\partial \phi(Z)}{\partial \tilde x_j}\right)^T\hat W
$$

在所考察区域内是 Lipschitz 的，即存在常数 $L_{\phi,j}>0$ 使得

$$
\left\|
\left(\frac{\partial \phi(\bar Z_j;m_j)}{\partial \tilde x_j}\right)^T\hat W
-
\left(\frac{\partial \phi(Z)}{\partial \tilde x_j}\right)^T\hat W
\right\|
\le
L_{\phi,j}\|\bar Z_j-Z\|
=
L_{\phi,j}\|e_j^Z\|,
$$

则仍由 $\tanh$ 的 $1$-Lipschitz 性得

$$
\|u_j^{p,\mathrm{exec}}-u_j^{p,\mathrm{cen}}\|
\le
\frac{1}{2}\|R_1^{-1}\|\bar g_p L_{\phi,j}\|e_j^Z\|.
$$

记

$$
c_{\gamma,j}=\frac{1}{2}\|R_1^{-1}\|\bar g_p L_{\phi,j},
$$

则

$$
\|u_j^{p,\mathrm{exec}}-u_j^{p,\mathrm{cen}}\|
\le
c_{\gamma,j}\|e_j^Z\|.
$$

### 7.2 中断阶段的有界扰动解释

因此，通信中断造成的额外影响可以视为一个由估计误差驱动的有界扰动项

$$
d_{\gamma}(Z,e^Z),
$$

并满足

$$
\|d_{\gamma}(Z,e^Z)\|
\le
\sum_{j=1}^{n} c_{\gamma,j}\|e_j^Z\|.
$$

如果通信中断窗口有限，且状态在该窗口内保持有界，则 $e_j^Z$ 有界，从而 $d_{\gamma}$ 有界。

### 7.3 恢复通信后的重新收敛

一旦通信恢复，根据前面的一步一致性结论，有

$$
e_j^Z(t_r^+)=0,\qquad \forall j.
$$

于是所有由估计误差引起的额外控制扰动在恢复后的第一次同步之后立刻消失。此时系统重新退化为全通信近似闭环系统，因此重新进入前述 UUB 框架。

**定理 3**

若近似全通信闭环系统一致最终有界，通信中断时间有限，且恢复后本地估计在一步内重新同步，则系统在断联期间保持有界，并在通信恢复后重新进入全通信 UUB 邻域。

**证明**

断联期间，闭环系统等价于全通信近似闭环系统加上有界扰动 $d_{\gamma}$。由 UUB 系统对有界扰动的鲁棒性，可知状态有界。恢复后，根据一致性命题，估计误差在一步内变为零，因此扰动项 $d_{\gamma}$ 消失，系统重新回到全通信近似闭环系统，故重新进入原 UUB 邻域。

---

## 8. 合理性自审

为避免理论结论写得过强，这里明确说明：

1. 对当前模型，最稳妥的主结论是 **一致最终有界**，而不是无条件的全局渐近稳定。
2. 若想得到更强的渐近稳定结论，需要进一步要求运行代价在闭环下对误差状态全局正定，或者要求逃避者项在稳定性估计中不再留下常值上界。
3. 对通信恢复后的“一步一致性”，这是一个非常强且合理的结论，因为它直接来自当前本地估计更新律，而不是额外假设。
4. 对中断期间的稳定性，本质上依赖于“控制误差对估计误差的 Lipschitz 界”和“中断时间有限”，因此把它解释为有界扰动作用下的 UUB 是合理的。

因此，这份推导与当前 $n$ 追一通信感知组级追捕模型是匹配的，不会出现“理论说得太强而实际系统并不满足”的问题。

补充说明：当前工程中的数值实验仍然使用 $n=3$ 的特例，但上述推导本身针对的是一般的 $n$ 追一场景。
