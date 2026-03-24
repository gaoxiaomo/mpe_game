# MPE 复现与改进思路: 详细推导与符号解释

本文档先给出当前复现代码对应的“论文基线推导”，再给出可作为论文创新点的“通信协同改进推导”。

目标是做到两件事:

1. 每个式子都说明“它在算什么、为什么这样写”。
2. 给出可直接落地到代码的改进公式，而不是只停留在概念。

---

## 1. 问题与符号定义

### 1.1 多追逐者-多逃避者模型

- 追逐者集合: \(\mathcal{P}=\{1,\dots,N_p\}\)
- 逃避者集合: \(\mathcal{E}=\{1,\dots,N_e\}\)
- 分配映射: \(\sigma(j)\in\mathcal{E}\)，表示追逐者 \(j\) 当前被分配去追哪一个逃避者

### 1.2 单个飞行器状态

对任一智能体（追逐者或逃避者），状态写成 6 维:

\[
x = [p_x,p_y,p_h,v_x,v_y,v_h]^T
\]

- \(p_x,p_y\): 平面位置
- \(p_h\): 高度（你前面说的 \(h\) 维）
- \(v_x,v_y,v_h\): 对应速度分量

控制输入:

\[
u=[u_x,u_y,u_h]^T
\]

每个输入都受到幅值约束（饱和）:

\[
\|u\|_{\infty}\le \bar u
\]

其中追逐者上限 \(\bar u_p\)，逃避者上限 \(\bar u_e\)。

### 1.3 期望位移（论文里的 \(r\)）

对“追逐者 \(j\) 追逃避者 \(i\)”这一对，定义期望位移 \(r_{j,i}\in\mathbb{R}^6\)。

相对误差（论文里常记为 \(h_{j,i}\)）:

\[
h_{j,i}=x_j^p-x_i^e+r_{j,i}
\]

含义:

- 若 \(h_{j,i}=0\)，表示追逐者 \(j\) 到达了“相对逃避者 \(i\) 的期望编队位置”；
- 不是简单“位置重合”，而是考虑了你要求的“保持一定相对距离/队形”。

---

## 2. 动力学与误差系统

### 2.1 单体动力学（通用写法）

论文与代码都可写成控制仿射形式:

\[
\dot x = f(x)+g(x)u
\]

其中:

- \(f(x)\): 自然动力学（气动、重力、阻尼等汇总）
- \(g(x)\): 控制通道矩阵

### 2.2 相对误差动力学

对追逐者 \(j\) 与其分配目标 \(i=\sigma(j)\):

\[
\dot h_j=\dot x_j^p-\dot x_i^e
=f_j^p-f_i^e+g_j^p u_j^p-g_i^e u_i^e
\]

含义:

- 该式告诉我们: 误差变化由双方动力学共同决定；
- 后续优化就是通过 \(u_j^p,u_i^e\) 让 \(h_j\) 变小或变大（博弈）。

---

## 3. 性能指标（代价函数）

## 3.1 基线 pursuer-centric 指标

对每个追逐者 \(j\)，定义无限时域指标:

\[
J_j=\int_0^\infty \Big(
h_j^TQh_j + \mathcal U_p(u_j^p)-\mathcal U_e(u_{\sigma(j)}^e)
\Big)\,dt
\]

含义逐项解释:

- \(h_j^TQh_j\): 状态误差惩罚（越远越差）
- \(\mathcal U_p(u_j^p)\): 追逐者控制努力代价（控制太大要付代价）
- \(-\mathcal U_e(\cdot)\): 逃避者对抗项（从追逐者视角，逃避者越“有效”越不利）

### 3.2 非二次饱和控制代价（关键）

为得到论文 Eq.(40) 的 `tanh` 控制律，采用非二次输入代价:

\[
\mathcal U_p(u)
=\sum_{k=1}^{3}
2\bar u_p r_{1,k}
\left(
u_k \operatorname{artanh}\!\left(\frac{u_k}{\bar u_p}\right)
+\frac{\bar u_p}{2}\ln\!\left(1-\left(\frac{u_k}{\bar u_p}\right)^2\right)
\right)
\]

\[
\mathcal U_e(u)
=\sum_{k=1}^{3}
2\bar u_e r_{2,k}
\left(
u_k \operatorname{artanh}\!\left(\frac{u_k}{\bar u_e}\right)
+\frac{\bar u_e}{2}\ln\!\left(1-\left(\frac{u_k}{\bar u_e}\right)^2\right)
\right)
\]

这个形式的意义:

- 对小输入近似二次型；
- 当 \(|u_k|\to \bar u\) 时，代价梯度会快速增大；
- 与 `tanh/artanh` 互为反函数，能得到闭式饱和控制。

对应导数（后面推导必用）:

\[
\frac{\partial \mathcal U_p}{\partial u_k}
=2\bar u_p r_{1,k}\operatorname{artanh}\!\left(\frac{u_k}{\bar u_p}\right)
\]

\[
\frac{\partial \mathcal U_e}{\partial u_k}
=2\bar u_e r_{2,k}\operatorname{artanh}\!\left(\frac{u_k}{\bar u_e}\right)
\]

---

## 4. HJI/HJB 与最优控制律推导

## 4.1 价值函数

对追逐者 \(j\)，定义价值函数:

\[
V_j(h_j)=\min_{u_j^p}\max_{u_{\sigma(j)}^e}
\int_t^\infty \Big(
h_j^TQh_j + \mathcal U_p(u_j^p)-\mathcal U_e(u_{\sigma(j)}^e)
\Big)d\tau
\]

含义:

- 追逐者要“最小化”未来总代价；
- 逃避者要“最大化”该代价（零和对抗）。

### 4.2 Hamiltonian

\[
\mathcal H_j
=h_j^TQh_j
+\mathcal U_p(u_j^p)-\mathcal U_e(u_{\sigma(j)}^e)
+(\nabla V_j)^T\!\left(
f_j^p-f_{\sigma(j)}^e+g_j^pu_j^p-g_{\sigma(j)}^eu_{\sigma(j)}^e
\right)
\]

HJI 方程:

\[
0=\min_{u_j^p}\max_{u_{\sigma(j)}^e}\mathcal H_j
\]

### 4.3 对追逐者输入求驻点

对 \(u_j^p\) 求偏导并设为 0:

\[
\frac{\partial \mathcal H_j}{\partial u_j^p}
=2\bar u_pR_1\operatorname{artanh}\!\left(\frac{u_j^p}{\bar u_p}\right)
+(g_j^p)^T\nabla V_j
=0
\]

因此:

\[
\operatorname{artanh}\!\left(\frac{u_j^p}{\bar u_p}\right)
=-\frac{1}{2\bar u_p}R_1^{-1}(g_j^p)^T\nabla V_j
\]

两边取 `tanh`:

\[
u_j^{p*}
=-\bar u_p\tanh\!\left(
\frac{1}{2\bar u_p}R_1^{-1}(g_j^p)^T\nabla V_j
\right)
\]

这就是论文 Eq.(40) 的核心结构（代码里已按这类形式实现）。

### 4.4 对逃避者输入求驻点

同理得到:

\[
u_i^{e*}
=-\bar u_e\tanh\!\left(
\frac{1}{2\bar u_e}R_2^{-1}(g_i^e)^T\nabla V_j
\right)
\]

说明:

- 这是“站在追逐者价值函数 \(V_j\) 视角”的逃避者最优反应；
- 因此 3v1 虽然只有 1 个真实逃避者，也会出现 3 套与 \(j\) 相关的 critic 权重 \(W_{j,s}\)。

---

## 5. 神经网络近似与离线差分最小二乘

## 5.1 价值函数近似

\[
V_j(h_j)\approx \hat V_j(h_j)=\hat W_j^T\phi(h_j)
\]

- \(\phi(\cdot)\): 特征向量（代码中 6 维特征）
- \(\hat W_j\): 第 \(j\) 个 critic 权重

梯度:

\[
\nabla \hat V_j(h_j)=\left(\frac{\partial \phi}{\partial h_j}\right)^T\hat W_j
\]

### 5.2 离线数据采样（固定策略滚动）

在第 \(s\) 次策略迭代，保持当前策略不变，跑很多时间步，收集:

\[
\{h_k,\;h_{k+1},\;r_k\}
\]

其中 \(r_k\) 是即时 stage cost。

### 5.3 差分 Bellman 方程

\[
\hat V_j(h_{k+1})-\hat V_j(h_k)\approx -r_k\Delta t
\]

代入线性近似:

\[
\big(\phi(h_{k+1})-\phi(h_k)\big)^T\hat W_j=-r_k\Delta t
\]

记:

\[
a_k^T=\phi(h_{k+1})-\phi(h_k),\quad b_k=-r_k\Delta t
\]

堆叠得到线性系统:

\[
A\hat W_j=b
\]

### 5.4 正则最小二乘（带先验）

\[
\hat W_j^{LS}
=\arg\min_W \|AW-b\|_2^2+\lambda\|W-\hat W_{j,s}\|_2^2
\]

闭式解:

\[
\hat W_j^{LS}
=(A^TA+\lambda I)^{-1}(A^Tb+\lambda\hat W_{j,s})
\]

然后做平滑更新:

\[
\hat W_{j,s+1}
=\hat W_{j,s}+\alpha_i(\hat W_j^{LS}-\hat W_{j,s})
\]

含义:

- \(\alpha_i\) 小（如 0.01）可避免权重跳变；
- 这就是你说的“离线策略 + 差分最小二乘 + 带入下一步继续求解”。

---

## 6. 为什么 3v1 有 3 个网络（而不是 1 个）

虽然是 3 个追逐者追 1 个逃避者，但每个追逐者 \(j\) 有不同:

1. 相对误差 \(h_j\)（由各自状态与各自 \(r_{j,1}\) 决定）
2. 动力学通道 \(g_j^p\)
3. 价值函数 \(V_j\)（“这个追逐者视角”的任务）

因此自然对应 3 个 critic:

\[
(\hat W_{1,s},\hat W_{2,s},\hat W_{3,s})
\]

这与论文中 Fig.4 给三条权重曲线是一致的。

---

## 7. 创新方向 A: 引入“追逐者间通信协同”到性能指标

这是最贴合你导师建议、且推导最自然的一种改法。

## 7.1 新增通信图

定义追逐者通信图:

\[
\mathcal G_p=(\mathcal P,\mathcal E_p),\quad
\mathcal N_j=\{k\mid (j,k)\in\mathcal E_p\}
\]

\(\mathcal N_j\) 是追逐者 \(j\) 能通信的邻居。

### 7.2 增广指标

把“队形协同”和“控制协同”加进 \(J_j\):

\[
J_j^{com}=\int_0^\infty
\Big[
h_j^TQh_j
+\mathcal U_p(u_j)-\mathcal W_e(u_{\sigma(j)}^e)
+\lambda_x\sum_{k\in\mathcal N_j}\|p_j-p_k-d_{jk}\|^2
+\lambda_u\sum_{k\in\mathcal N_j}\|u_j-u_k\|^2
\Big]dt
\]

各项含义:

- \(\lambda_x\): 空间协同强度（保证队形/覆盖）
- \(\lambda_u\): 控制协同强度（稳定时最小化，也就是控制输入相同，队形不变）
- \(d_{jk}\): 期望队形偏移

### 7.3 新 Hamiltonian 对 \(u_j\) 的驻点

\[
\frac{\partial \mathcal H_j^{com}}{\partial u_j}
=2\bar u_pR_1\operatorname{artanh}\!\left(\frac{u_j}{\bar u_p}\right)
+(g_j^p)^T\nabla V_j
+2\lambda_u\sum_{k\in\mathcal N_j}(u_j-u_k)
=0
\]

得到隐式最优控制:

\[
u_j^*
=-\bar u_p\tanh\!\left(
\frac{1}{2\bar u_p}
R_1^{-1}
\left[
(g_j^p)^T\nabla V_j
+2\lambda_u\sum_{k\in\mathcal N_j}(u_j-u_k)
\right]
\right)
\]

该式“隐式”的原因:

- 右侧还有 \(u_j\) 本身（在协同项里），不能一步闭式直接解。

### 7.4 可实现的显式近似（工程落地）

用上一时刻输入近似协同项:

\[
u_j[k]
=-\bar u_p\tanh\!\left(
\frac{1}{2\bar u_p}R_1^{-1}
\left[
(g_j^p)^T\nabla V_j[k]
+2\lambda_u\sum_{k\in\mathcal N_j}(u_j[k-1]-u_k[k-1])
\right]
\right)
\]

意义:

- 保持 Eq.(40) 结构；
- 只额外引入邻居控制消息；
- 非常容易改进到现有代码。

---

## 8. 创新方向 B: 通信中断/恢复场景（更工程化）

定义链路可用变量:

\[
\gamma_{jk}(t)\in\{0,1\}
\]

- \(\gamma_{jk}=1\): 链路可用
- \(\gamma_{jk}=0\): 链路中断

把协同项改为:

\[
\sum_{k\in\mathcal N_j}\gamma_{jk}(t)\|u_j-u_k\|^2
\]

则控制律自动变为:

\[
u_j[k]
=-\bar u_p\tanh\!\left(
\frac{1}{2\bar u_p}R_1^{-1}
\left[
(g_j^p)^T\nabla V_j[k]
+2\lambda_u\sum_{k\in\mathcal N_j}\gamma_{jk}[k](u_j[k-1]-u_k[k-1])
\right]
\right)
\]

含义:

- 断链时协同自动减弱；
- 恢复后自动回到协同追捕；
- 很适合你导师提到的“追丢后再追上”实验叙事。

---

## 9. 创新方向 C: 接力追捕（leader-follower）

定义领导者索引 \(\ell(t)\in\{1,2,3\}\)。

### 9.1 leader 指标

\[
J_{\ell}=\int \big(h_{\ell}^TQh_{\ell}+\mathcal U_p(u_{\ell})-\mathcal U_e(u_{\sigma(\ell)}^e)\big)\,dt
\]

### 9.2 follower 指标

\[
J_j^{fol}
=\int
\Big(
\|x_j^p-x_{\ell}^p-r_{j\ell}\|_{Q_f}^2
+\lambda_u\|u_j-u_{\ell}\|^2
+\mathcal U_p(u_j)
\Big)\,dt,\quad j\neq \ell
\]

意义:

- 只有 leader 强对抗逃避者；
- 其余两机保持编队/支援；
- 可按时间或事件切换 \(\ell(t)\)（接力）。

---

## 10. 自动交换（非预设时刻）建议公式

你强调“必须自动交换”，最稳妥写法是把交换变成每个时刻的判据优化。

### 10.1 代价矩阵

\[
d_{j,i}(t)=\left\|\nu\odot\big(x_j^p-x_i^e+r_{j,i}\big)\right\|
\]

- \(\nu\): 各状态通道权重（可强调 XY 或 H）
- \(d_{j,i}\): 当前“追逐者 \(j\) 追逃避者 \(i\)”的代价

### 10.2 带切换惩罚的分配优化

\[
A(t)=\arg\min_{A\in\Pi}
\sum_{j=1}^{N_p}\sum_{i=1}^{N_e}a_{j,i}d_{j,i}(t+\tau)
+\beta_{sw}\|A-A(t-\Delta t)\|_1
\]

约束 \(\Pi\):

- 每个追逐者只分配一个目标: \(\sum_i a_{j,i}=1\)
- 每个逃避者至多一个追逐者（或等人数时恰好一个）

含义:

- 第一项: 追更近/更有利目标（可加 lookahead \(\tau\)）
- 第二项: 避免抖动式频繁切换

这是“自发触发”，不是预设时间点触发。

---

## 11. 你最关心的 \(u_p\) / \(u_e\) 饱和比较如何正确表达

你前面提到的核心是对的: 关键是“控制是否真的触发到饱和边界”。

应该检查三件事:

1. 原始 `tanh` 内部量 \(\rho\) 是否足够大（否则 \(\tanh(\rho)\approx \rho\)，永远小幅输入）。
2. 实际施加输入是否经过 `clip` 到 \([-\bar u,\bar u]\)。
3. 逃避者策略是否对抗性足够强（否则 \(u_e\) 太小会导致任何 \(\bar u_p\) 都能追上）。

论文式表达不是“仅比较上限数字”，而是“在可比对抗策略下比较捕获结果/时间”。

---

## 12. 建议的论文创新主线（可直接写在你论文里）

建议你用“通信协同 + 中断恢复鲁棒性”作为创新:

1. **基线**: V-SNAC 追逃（复现论文）。
2. **改进1**: 在性能指标加入追逐者间控制协同项 \(\|u_i-u_j\|^2\)。
3. **改进2**: 引入链路变量 \(\gamma_{ij}(t)\) 建模通信丢包/恢复。
4. **改进3**: 分配层采用“自发判据交换”而非预设交换时刻。

这样创新点清晰、推导连贯、实验可验证。

---

## 13. 下一步实现映射（我后续可直接按此改代码）

1. 在 `controller.py` 加入通信协同显式近似控制律（保留 Eq.(40) 主体）。
2. 在 `simulator.py` 增加 \(\gamma_{ij}(t)\) 场景调度（中断/恢复）。
3. 在 `graph_switch.py` 改为“代价最小 + 切换惩罚”的自动交换。
4. 在 `plotting.py` 增加:
   - 通信可用率曲线
   - 捕获时间对比
   - 切换次数/切换时刻
5. 在 `REPORT.md` 自动写出:
   - \(u_p>\!u_e,\;u_p=\!u_e,\;u_p<\!u_e\) 的捕获结论
   - 中断期间误差上升与恢复后收敛对比

---

如果你认可这个文档结构，我下一步就按第 13 节把代码改成“可跑的创新版”，并给你一份对应的实验脚本和图表说明。
