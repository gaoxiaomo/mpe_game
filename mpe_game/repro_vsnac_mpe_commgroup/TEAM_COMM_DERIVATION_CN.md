# 通信感知组级追捕模型推导与稳定性说明

## 1. 目标与建模范围

本文档对应目录 [repro_vsnac_mpe_commgroup](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup) 中实现的**三追一通信感知组级追捕版本**。

这一版的出发点是：

- 原复现版在 `3v1` 场景中，本质上仍是“3 个 pursuer critic 并行追同一个 evader”
- 但如果三个追逐者的目标相同，那么更自然的建模方式不是各自最优，而是把三者看成一个**追捕小组**
- 通信的意义也从“换目标”转移为“共享组状态、协同决策、在丢包或断联时退化为局部估计控制”

因此，这里采用：

1. 一个 **team critic** 代替 3 个独立 pursuer critic
2. 训练时使用完整组状态
3. 执行时，每个 pursuer 使用按通信矩阵掩码后的组观测 \(\bar Z_j=\Gamma_j Z\)
4. 被屏蔽的队友状态块在当前时刻直接置零，并通过可见性掩码同步关闭对应特征与耦合项
5. 通信恢复时，\(\Gamma_j\) 回到全可见，本地观测立即重新同步到真实组状态

需要特别说明的是：

- 理论推导仍保留组级追逃博弈的 Hamiltonian / HJI 结构
- 但在当前实验实现中，为了单独观察“追逐者通信”本身的影响，逃避者实际执行的是**同一条有界机动轨迹**，而不是两组实验中各自变化的对抗策略
- 因此，实验部分更接近“组级最优追捕 + 有界扰动 evader”的设置

这一步是刻意设计的，因为它能把通信丢失/恢复对 `Eteam` 的影响单独拉出来看。

---

## 2. 单体动力学与相对误差状态

每个追逐者和逃避者都沿用原工程中的 6 维非线性飞行动力学：

\[
\dot x = f(x) + g(x)u
\]

其中：

\[
x = [x, y, h, v_x, v_y, v_h]^T \in \mathbb R^6,
\qquad
u = [u_1, u_2, u_3]^T \in \mathbb R^3
\]

对第 \(j\) 个追逐者：

\[
\dot x_j^p = f_j(x_j^p) + g_j(x_j^p)u_j^p
\]

对唯一逃避者：

\[
\dot x^e = f_e(x^e) + g_e(x^e)u^e
\]

当前代码对应实现：

- 动力学：[dynamics.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/dynamics.py)
- 组控制器：[team_comm_controller.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_controller.py)

### 2.1 追逃相对误差

对每个追逐者定义相对误差：

\[
\tilde x_j = x_j^p - x^e + r_j, \qquad j=1,2,3
\]

其中 \(r_j\) 是期望位移。

它的动力学为：

\[
\dot{\tilde x}_j
=
f_j(x_j^p)-f_e(x^e)
+
g_j(x_j^p)u_j^p
-
g_e(x^e)u^e
+
\dot r_j
\]

若 \(r_j\) 为常数，则 \(\dot r_j = 0\)。

---

## 3. 组级堆叠状态与 team objective

### 3.1 组级状态

把三个追逐者的相对误差堆叠为一个组状态：

\[
Z = \operatorname{col}(\tilde x_1,\tilde x_2,\tilde x_3) \in \mathbb R^{18}
\]

这一步是新模型的核心。原来的 `3v1` 是三个局部 critic；现在是一个组级 critic。

### 3.2 组级性能指标

定义追捕小组 \(\mathcal S = \{1,2,3\}\)，组级运行代价为：

\[
\ell(Z,U^p,u^e)
=
\sum_{j=1}^{3} P_j(\tilde x_j)
+
\sum_{j=1}^{3} U(u_j^p)
-
W(u^e)
\]

于是组级性能指标为：

\[
J_{\mathcal S}(Z(0),U^p,u^e)
=
\int_0^{\infty} \ell(Z,U^p,u^e)\,dt
\]

其中：

\[
U^p = \operatorname{col}(u_1^p, u_2^p, u_3^p)
\]

并且：

- \(P_j(\tilde x_j)\)：第 `j` 个追逐者相对误差代价
- \(U(u_j^p)\)：追逐者输入代价
- \(W(u^e)\)：逃避者输入代价

这里的含义是：

- 三个 pursuer 共同最小化同一个组级代价
- evader 在理论上试图增大这个代价

从而形成一个组级 min-max 博弈。

---

## 4. 组级动力学紧凑形式

将三个误差方程堆叠，可写为：

\[
\dot Z = F(Z) + G(Z)U^p + E(Z)u^e
\]

其中：

\[
F(Z)=
\operatorname{col}
\Big(
 f_1(x_1^p)-f_e(x^e)+\dot r_1,
 f_2(x_2^p)-f_e(x^e)+\dot r_2,
 f_3(x_3^p)-f_e(x^e)+\dot r_3
\Big)
\]

\[
G(Z)=
\operatorname{diag}(g_1(x_1^p), g_2(x_2^p), g_3(x_3^p))
\]

\[
E(Z)= - \operatorname{col}(g_e(x^e), g_e(x^e), g_e(x^e))
\]

这个形式使得后面的 Hamiltonian 写法很直接。

---

## 5. 组级 Hamiltonian 与 HJI 方程

定义组级值函数：

\[
V(Z)=\min_{U^p}\max_{u^e} J_{\mathcal S}(Z,U^p,u^e)
\]

则 Hamiltonian 为：

\[
H(Z,U^p,u^e,\nabla V)
=
\ell(Z,U^p,u^e)
+
\nabla V(Z)^T \big(F(Z)+G(Z)U^p+E(Z)u^e\big)
\]

即：

\[
H
=
\sum_{j=1}^{3} P_j(\tilde x_j)
+
\sum_{j=1}^{3} U(u_j^p)
-
W(u^e)
+
\nabla V(Z)^T \big(F(Z)+G(Z)U^p+E(Z)u^e\big)
\]

对应 HJI 方程为：

\[
0=
\min_{U^p}\max_{u^e} H(Z,U^p,u^e,\nabla V)
\]

这是本通信版的组级 Hamiltonian。

---

## 6. 最优控制结构

### 6.1 pursuer 侧

对每个追逐者 \(u_j^p\) 做驻值条件，可以得到与原论文 Eq.(40) 同型的饱和控制结构：

\[
u_j^{p*}
=
-\bar u_p
\tanh\left(
\frac{1}{2\bar u_p}
R_1^{-1}
g_j(x_j^p)^T
\frac{\partial V}{\partial \tilde x_j}
\right)
\]

注意这里与原论文最大的区别在于：

- 原论文中，\(\partial V / \partial \tilde x_j\) 来自单个 pursuer critic
- 这里，\(\partial V / \partial \tilde x_j\) 来自同一个 team critic 的第 `j` 个分块梯度

所以每个 pursuer 的控制已经不是“只看自己”，而是隐含地受完整组状态影响。

### 6.2 evader 侧

理论上，对逃避者做极大化，可得：

\[
u^{e*}
=
-\bar u_e
\tanh\left(
\frac{1}{2\bar u_e}
R_2^{-1}
g_e(x^e)^T
\sum_{j=1}^{3}\frac{\partial V}{\partial \tilde x_j}
\right)
\]

这表示 evader 面对的是整组 pursuer 的联合压力。

### 6.3 当前实验实现与理论控制的区别

在当前实验代码中，为了隔离通信影响：

- **训练时**，保留组级 critic 与组级 Hamiltonian 的结构
- **执行时**，evader 实际采用固定的有界机动轨迹

记理论 evader 控制为 \(u^{e,\mathrm{virtual}}\)，实际执行控制为 \(u^{e,\mathrm{applied}}\)，则实验中：

\[
u^{e,\mathrm{applied}}(t)=d_e(t)
\]

其中 \(d_e(t)\) 为有界时变机动输入。

因此闭环系统更准确地说是：

\[
\dot Z = F(Z) + G(Z)U^p + E(Z)d_e(t)
\]

它是一个带有界外扰的组级追捕系统。

---

## 7. Team critic 近似结构

当前实现采用线性参数化值函数：

\[
\hat V(Z)=\hat W^T \phi(Z)
\]

其中 \(\hat W\in\mathbb R^{27}\)。

### 7.1 局部特征

对每个 pursuer 的误差块保留 6 个局部特征：

\[
\phi_j^{\mathrm{local}}
=
\begin{bmatrix}
p_{jx}v_{jx}\\
p_{jy}v_{jy}\\
p_{jh}v_{jh}\\
\frac12 v_{jx}^2\\
\frac12 v_{jy}^2\\
\frac12 v_{jh}^2
\end{bmatrix}
\]

三个 pursuer 一共给出 18 项。

### 7.2 组间耦合特征

为了让通信对控制真的有影响，还加入了三对 pursuer 之间的耦合项：

\[
\phi_{ab}^{\mathrm{cross}}
=
\begin{bmatrix}
(p_{ax}-p_{bx})(v_{ax}-v_{bx})\\
(p_{ay}-p_{by})(v_{ay}-v_{by})\\
(p_{ah}-p_{bh})(v_{ah}-v_{bh})
\end{bmatrix},
\quad (a,b)\in\{(1,2),(1,3),(2,3)\}
\]

共 9 项。

因此总特征为：

\[
\phi(Z) \in \mathbb R^{27}
\]

其代码实现：

- [team_comm_features.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_features.py)

### 7.3 可见性掩码下的特征退化

对第 \(j\) 个 pursuer，定义可见性向量与块对角掩码矩阵：

\[
m_j =
\begin{bmatrix}
\gamma_{1j}\\
\gamma_{2j}\\
\gamma_{3j}
\end{bmatrix},
\qquad
\Gamma_j = \operatorname{diag}(\gamma_{1j}I_6,\gamma_{2j}I_6,\gamma_{3j}I_6)
\]

其中 \(\gamma_{kj}\in\{0,1\}\) 表示第 \(k\) 个状态块在时刻 \(t\) 是否对第 \(j\) 个 pursuer 可见。

执行时，真正送入 critic 的不是完整组状态 \(Z\)，而是掩码后的组观测：

\[
\bar Z_j
=
\Gamma_j Z
=
\operatorname{col}(\gamma_{1j}\tilde x_1,\gamma_{2j}\tilde x_2,\gamma_{3j}\tilde x_3)
\]

于是当前实现对应的特征映射可写成：

\[
\phi(\bar Z_j;m_j)
\]

并且满足：

\[
\phi_{k,j}^{\mathrm{local}}(\bar Z_j;m_j)
=
\gamma_{kj}\,\phi_k^{\mathrm{local}}(Z)
\]

\[
\phi_{ab,j}^{\mathrm{cross}}(\bar Z_j;m_j)
=
\gamma_{aj}\gamma_{bj}\,\phi_{ab}^{\mathrm{cross}}(Z)
\]

也就是说，隐藏块不会引入历史保持项，而是在当前时刻被直接零填充；同时，任何涉及隐藏队友的局部项和交叉项都会被掩码一起压掉。这与 [team_comm_features.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_features.py)、[team_comm_controller.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_controller.py) 和 [team_comm_simulator.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_simulator.py) 的实现一致。

---

## 8. 通信中断下的掩码观测控制

### 8.1 掩码组观测

训练时，每个 pursuer 都使用完整真实组状态：

\[
\bar Z_j = Z,
\qquad
m_j = \mathbf 1
\]

执行时，对第 `j` 个 pursuer：

\[
\bar Z_j
=
\Gamma_j Z
=
\operatorname{col}(\gamma_{1j}\tilde x_1,\gamma_{2j}\tilde x_2,\gamma_{3j}\tilde x_3)
\]

其中：

- 自身块始终可测，因此 \(\gamma_{jj}=1\)
- 若链路可用，则 \(\gamma_{kj}=1\)，第 \(k\) 个块按真值进入 \(\bar Z_j\)
- 若链路断开，则 \(\gamma_{kj}=0\)，第 \(k\) 个块在当前时刻直接被置零

即：

\[
\bar{\tilde x}_{k|j}(t)=
\begin{cases}
\tilde x_k(t), & \gamma_{kj}(t)=1,\\
\mathbf 0, & \gamma_{kj}(t)=0.
\end{cases}
\]

其中 \(\gamma_{kj}(t)\in\{0,1\}\) 表示通信可用性。这正是当前代码中 “zero-fill + visibility mask” 的观测模型。

### 8.2 掩码观测驱动的 pursuer 控制

因此 pursuer 实际执行控制为：

\[
u_j^{p}
=
-\bar u_p
\tanh\left(
\frac{1}{2\bar u_p}
R_1^{-1}
g_j(x_j^p)^T
\frac{\partial \hat V(\bar Z_j;m_j)}{\partial \tilde x_j}
\right)
\]

在线性参数化实现里，上式等价于：

\[
\frac{\partial \hat V(\bar Z_j;m_j)}{\partial \tilde x_j}
=
\left(\frac{\partial \phi(\bar Z_j;m_j)}{\partial \tilde x_j}\right)^T\hat W
\]

这就是“集中训练、分散执行、通信中断导致掩码观测失真”的数学表达。

### 8.3 与全通信控制的偏差

定义第 \(j\) 个 pursuer 的掩码观测误差：

\[
\eta_j = \bar Z_j - Z = -(I-\Gamma_j)Z
\]

若闭环区域内特征映射与 \(\tanh(\cdot)\) 控制律局部 Lipschitz，则存在常数 \(L_u>0\)，使得：

\[
\|u_j^{p,\mathrm{mask}}-u_j^{p,\mathrm{full}}\|
\le
L_u \|\eta_j\|
\]

因此，通信丢失在当前实现里对应的是**掩码观测诱导的有界控制扰动**，而不是“旧估计值随时间漂移”的误差模型。

---

## 9. 训练更新与收敛说明

当前实现中，critic 使用 Bellman difference 最小二乘更新。样本满足：

\[
\hat W^T \phi(Z_{k+1}) - \hat W^T \phi(Z_k) \approx -\ell_k \Delta t
\]

写成线性方程：

\[
A\hat W \approx b
\]

其中每一行对应：

\[
a_k = \phi(Z_{k+1}) - \phi(Z_k),
\qquad
b_k = -\ell_k\Delta t
\]

然后求解带 ridge 的正则化最小二乘：

\[
\hat W^+
=
\arg\min_W \|AW-b\|^2 + \lambda \|W-W_{\mathrm{prev}}\|^2
\]

这一步在当前通信版代码里由 [team_comm_ls.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_ls.py) 完成。

在当前通信版实现里，又做了三点数值稳定化扩展：

- 对局部项权重与交叉项权重分别设置投影范围，而不再沿用单 critic 的统一正区间裁剪
- 每一轮策略评估只使用**当前策略 freshly generated 的样本集**，不把旧策略样本跨轮累积混合
- 每轮 LS 解之后，不直接全量替换，而是做**投影阻尼更新 + 回溯接受**：只有当权重能量 \(\|\hat W\|^2\) 不上升、且验证指标不明显恶化时才接受该步

因此，当前实现中的“收敛”更准确地说是：

- 对每一轮 ridge LS 子问题，唯一解严格存在
- 对每一轮 accepted update，权重能量 \(\|\hat W_s\|^2\) 单调不增
- 在探索逐渐减小、样本激励充分、LS 解变化逐渐减小的条件下，原始训练权重序列本身收敛到一个稳定邻域，而不是只靠 checkpoint 事后挑选

### 命题 1：正则化最小二乘子问题的唯一性

若 \(\lambda>0\)，则矩阵 \(A^TA + \lambda I\) 正定，因此每一轮 LS 子问题存在唯一解。

**证明**：
对任意非零向量 \(z\)，有

\[
z^T(A^TA+\lambda I)z = \|Az\|^2 + \lambda \|z\|^2 > 0
\]

故 \(A^TA+\lambda I\) 正定，唯一解存在。

### 命题 2：在固定样本集上的单步更新是压缩到 ridge 解的

当前实现不是直接令 \(\hat W \leftarrow \hat W^+\)，而是：

\[
\hat W_{s+1} = \hat W_s + \alpha (\hat W^+_s - \hat W_s),
\qquad 0<\alpha\le 1
\]

若 \(\hat W^+_s\) 固定，则该更新对误差 \(e_s = \hat W_s - \hat W^+_s\) 满足：

\[
e_{s+1} = (1-\alpha)e_s
\]

故：

\[
\|e_{s+1}\| = |1-\alpha|\,\|e_s\| < \|e_s\|
\]

说明在固定目标解条件下，该步是严格收缩的。

### 命题 3：投影阻尼接受律保证权重能量单调不增

记本轮 ridge LS 解为 \(\hat W_s^+\)，候选更新为

\[
\hat W_{s+1}^{\mathrm{cand}}
=
\Pi_\Omega \Big(\hat W_s + \alpha_s(\hat W_s^+ - \hat W_s)\Big)
\]

其中：

- \(\Pi_\Omega\) 是对局部项与交叉项分别做有界投影
- \(\alpha_s>0\) 为阻尼步长
- 若候选不满足
\[
\|\hat W_{s+1}^{\mathrm{cand}}\|^2 \le \|\hat W_s\|^2 + \varepsilon
\]
则继续缩小 \(\alpha_s\) 回溯，直到满足条件或本轮拒绝更新

则对所有被接受的更新，有

\[
\|\hat W_{s+1}\|^2 \le \|\hat W_s\|^2 + \varepsilon
\]

在 \(\varepsilon\to 0\) 的实现近似下，可视作单调不增。

**证明**：

这是接受律本身的直接结果。因为只有满足上述不增条件的候选才会被写回为 \(\hat W_{s+1}\)，故对接受序列必然成立。又因为 \(\Omega\) 为紧集，\(\hat W_s\in\Omega\)，所以 \(\{\hat W_s\}\) 有界。

### 命题 4：在持续采样与持久激励下，权重序列收敛到一个有界邻域

设：

1. 组状态有界
2. 特征 \(\phi(Z)\) 和雅可比有界
3. 样本满足足够的持久激励，使得正规方程条件数不退化
4. 近似误差有界

则权重更新序列 \(\{\hat W_s\}\) 有界，并收敛到理想权重 \(W^*\) 的一个有界邻域。

并进一步假设：

5. 探索幅值 \(\sigma_s\) 随策略迭代逐渐减小
6. 每轮 LS 求解使用的是当前策略生成的新样本，因此 \(\hat W_s^+\) 是当前策略下的局部一致策略评估解

**证明思路**：

- 由命题 1，每轮 ridge LS 的目标解唯一且有界
- 由命题 2，更新是朝该唯一目标解的收缩步
- 由命题 3，训练过程中被接受的权重序列始终保持有界，且权重能量单调不增
- 当探索减小、每轮样本都来自当前策略、且样本满足激励条件时，\(\hat W_s^+\) 的轮间变化会逐渐减小
- 因此整体序列表现为“慢变目标上的受限收缩跟踪”
- 在近似误差和有限样本误差存在时，只能保证收敛到有界邻域，而不是严格收敛到精确最优权重

当前实现进一步通过回溯接受律，把这种“有界邻域收敛”落实成可观测的训练现象：整体 \(\|\hat W\|^2\) 曲线先下降、后趋于平稳，不再出现后期明显回升漂移。

---

## 10. 无通信中断时的闭环稳定性说明

### 10.1 理想值函数下的 Lyapunov 候选

在无中断、且使用理想值函数 \(V^*(Z)\) 时，可把 \(V^*(Z)\) 视为 Lyapunov 候选函数。若 HJI 方程成立，则沿最优闭环轨迹有：

\[
\dot V^*(Z)
=
\nabla V^{*T}(F+GU^{p*}+Eu^{e*})
=
-\ell(Z,U^{p*},u^{e*})
\]

若 \(\ell\) 正定，则 \(\dot V^*(Z)\le 0\)，系统在原点附近稳定。

### 10.2 近似值函数下的实际结论

由于实现中使用的是 \(\hat V\)，而不是 \(V^*\)，因此我们更合理的结论是**一致最终有界**（UUB）而不是严格渐近稳定。

### 定理 1：无中断闭环的 UUB 性质

设：

1. \(V^*(Z)\) 正定、径向无界
2. 逼近误差满足
   \[
   \|\nabla \hat V(Z)-\nabla V^*(Z)\| \le \varepsilon_V
   \]
3. 外部 evader 机动输入有界：
   \[
   \|d_e(t)\| \le \bar d
   \]
4. 闭环状态有界

则存在常数 \(c_1,c_2,c_3>0\)，使得：

\[
\dot V^*(Z)
\le
-c_1 \|Z\|^2 + c_2 \varepsilon_V^2 + c_3 \bar d^2
\]

因此闭环相对误差状态 \(Z\) 一致最终有界。

**证明思路**：

- 在理想最优控制下，\(\dot V^* = -\ell^*(Z)\)
- 用 \(\hat V\) 代替 \(V^*\) 相当于引入控制扰动项
- 有界 evader 机动轨迹引入额外有界扰动项
- 利用 Young 不等式可把扰动项上界为 \(c_2\varepsilon_V^2 + c_3\bar d^2\)
- 当 \(\|Z\|\) 足够大时，负定项仍占优，故得到 UUB

这就是本实现中“能追上并收敛，但不保证精确到零”的理论解释。

---

## 11. 通信中断/恢复下的估计误差与稳定性

定义本地掩码观测误差：

\[
\eta_j = \bar Z_j - Z
\]

### 11.1 通信正常时

当所有链路正常：

\[
\eta_j = 0
\]

### 11.2 通信中断时

若中断期间第 \(j\) 个 pursuer 只能看到部分状态块，则有：

\[
\eta_j(t) = (\Gamma_j(t)-I)Z(t)
\]

由于隐藏块在当前时刻被直接置零，因此有精确关系：

\[
\|\eta_j(t)\|^2
=
\sum_{k=1}^{3}\bigl(1-\gamma_{kj}(t)\bigr)\|\tilde x_k(t)\|^2
\]

从而可得：

\[
\|\eta_j(t)\|
\le
\|(I-\Gamma_j(t))Z(t)\|
\]

这说明：

- 被屏蔽的队友块越多，或被屏蔽块本身误差越大，则 \(\eta_j\) 越大
- 这与当前实验中 `estimate mismatch` 图一致

### 11.3 通信恢复时

在恢复时刻 \(t_r\)，若所有链路恢复，则 \(\Gamma_j(t_r)=I\)，因此：

\[
\eta_j(t_r^+) = 0
\]

这解释了为什么实验图中估计误差在恢复瞬间跳回 0。

### 定理 2：带有限时中断的 UUB

设：

1. 无中断闭环满足定理 1 的 UUB 条件
2. 通信中断区间长度有界，且恢复时间有限
3. 控制律对掩码观测误差是局部 Lipschitz 的
4. 估计误差满足 \(\|\eta_j(t)\|\le \bar \eta\)

则带通信中断/恢复的闭环系统仍然一致最终有界；并且在通信恢复后，系统重新回到无中断系统对应的 UUB 邻域。

**证明思路**：

- 通信中断导致 pursuer 控制偏差
  \[
  u_j^p(\bar Z_j;m_j)-u_j^p(Z)
  \]
- 由 Lipschitz 条件，有
  \[
  \|u_j^p(\bar Z_j;m_j)-u_j^p(Z)\| \le L_u \|\eta_j\|
  \]
- 因此中断相当于在无中断闭环上叠加有界输入扰动
- 由 ISS/UUB 标准结果，闭环状态保持有界
- 恢复时 \(\eta_j\to 0\)，系统重新退化为无中断闭环，因此重新回到无中断邻域

这与实验观察一致：

- `drop/recovery` 会让捕获变慢
- 但恢复后仍继续收敛并最终捕获

---

## 12. 关于 Eteam 与个体误差曲线的解释

当前主评价指标是：

\[
E_{\mathrm{team}}(t)
=
\sum_{j=1}^3 \|\nu \tilde x_j(t)\|,
\qquad
\nu=[1,1,1,0,0,0]
\]

只看 `x/y/h` 误差，不把速度误差直接算进最终展示指标。

需要强调：

- `drop/recovery` 不一定使每个 pursuer 在每个时刻都更差
- 因为 team critic 优化的是组级目标，而不是每个个体的瞬时贪心最优
- 因此可能出现“某一架机个体误差短时更小，但团队误差更大”的情况

所以这版结果里，真正应该优先看的，是：

1. `Eteam` 曲线
2. 捕获时间
3. 通信恢复后是否重新收敛

而不是单独某一架机的局部误差是否处处更优。

---

## 13. 为什么图上的 Eteam 尾部处理成 0

当前评估逻辑里，捕获的判据是：

\[
\max_j \|\nu \tilde x_j\| \le r_c
\]

其中 \(r_c\) 为捕获半径。

因此严格来说：

- 在捕获时刻，`Eteam` 一般不等于 0
- 它只表示已经进入捕获域

但为了和原复现图风格一致，当前展示图采用了**paper-style zero tail**：

- 到达捕获时刻后，后续展示部分补成 0
- 这只是绘图口径，表示“任务已完成”
- 真正数值比较仍然保存在 `summary.json` 里

也就是说：

- **图像尾部的 0 是任务完成标记**
- **不是说物理误差真的连续演化到了精确 0**

---

## 14. 代码映射

- 通信场景定义：
  [team_comm_config.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_config.py)

- 组级特征与梯度：
  [team_comm_features.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_features.py)

- 组级控制器与组代价：
  [team_comm_controller.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_controller.py)

- 通信中断/恢复执行逻辑：
  [team_comm_simulator.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_simulator.py)

- 图与结果输出：
  [team_comm_plotting.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/mpe_repro/team_comm_plotting.py)
  [run_team_comm.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/run_team_comm.py)

---

## 15. 这一版理论结论的边界

这份文档给出的稳定性/收敛性结论是**在标准假设下的半严格证明与工程化证明思路**，其边界如下：

- 对 ridge LS 子问题的唯一性，是严格成立的
- 对更新“收缩到本轮 LS 解”的结论，是严格成立的
- 对整体权重收敛、无中断闭环 UUB、以及中断/恢复后的重新收敛，属于建立在有界性、Lipschitz、激励条件、近似误差有界等标准假设上的理论结论
- 当前实验使用了固定有界 evader 机动轨迹，因此更准确地说，这是“组级 team critic pursuer + bounded evader disturbance”的闭环稳定分析

这和完全严格的 HJI 精确最优性证明不同，但已经与当前实际实现是一致的，不会出现“理论说一套，代码做另一套”的问题。

---

## 16. 后续可继续扩展的方向

在当前 team critic 基础上，最自然的下一步是把以下项继续并入组代价：

### 16.1 编队项

\[
\sum_{j<k}(x_j^p-x_k^p-d_{jk})^T Q_c (x_j^p-x_k^p-d_{jk})
\]

作用：

- 保持包围几何结构
- 避免某一架机贪心直冲、其余队员掉队

### 16.2 接力项 / 角色项

引入 leader / follower / reserve 角色变量，构造时间切换或能量相关代价项。

### 16.3 随机丢包而非整段中断

把 \(\gamma_{jk}(t)\) 从分段常数扩展为随机过程，更贴近实际链路。

这些都可以在当前 team-state 与 team critic 框架上继续加，而不需要推翻现有结构。
