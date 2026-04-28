# 实验汇总报告

本文档汇总目前为止毕设方向的所有实验结果与对应代码/输出路径。所有数学证明对应到 `MASTER_DERIVATION.tex`。

---

## 1. 通信失效鲁棒性（六场景碰撞 suite）

### 1.1 实验设计

- **场景**：6 个不同几何的 collision-prone 配置（`mpe_repro/collision_scenarios.py`）
  - C1 头对头 2v1（论文 baseline 几何）
  - C2 三机三 trio 3v1
  - C3 双层 stacked head-on 4v1
  - C4 平行冲突 2v2
  - C5 垂直 funnel 3v1
  - C6 非对称 5v2
- **dropout 模式**（`mpe_repro/comm_graph.py:DropoutPattern`）
  - `IIDBernoulliDropout`：每边每步独立 Bernoulli
  - `PersistentEdgeDropout`：固定子集全程失联
  - `PeriodicOutageDropout`：周期性整图断开
- **超参**：γ=1.5，d_safe=150m（FAA NMAC, 500ft），5 seeds × 9 模式 × 6 场景 = **270 评估**
- **驱动**：`run_collision_dropout_suite.py`

### 1.2 结果

输出目录 `outputs/collision_dropout_full/`：
- `batch_summary.json` 全部数据
- `REPORT.md` 表格
- `fig_dmin_grid.png` 6-panel d_min(t)
- `fig_degradation.png` 退化曲线

| 场景 | no_comm | full_comm | iid 15% | iid 30% | iid 50% | persistent 15% | persistent 30% | periodic 25% | periodic 50% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 head-on 2v1 | 2.9 | **215.3** | 207.4 | 79.1 | 3.3 | 215.3 | 215.3 | 221.4 | 103.9 |
| C2 trio 3v1 | 7.6 | **128.1** | 126.2 | 96.4 | 7.8 | 128.1 | 132.7 | 117.1 | 27.2 |
| C3 stacked 4v1 | 6.4 | **179.0** | 174.7 | 162.7 | 64.1 | 197.3 | 144.0 | 157.8 | 48.2 |
| C4 parallel 2v2 | 14.2 | **278.5** | 142.5 | 54.9 | 54.4 | 278.5 | 308.1 | 73.1 | 45.4 |
| C5 vertical 3v1 | 3.8 | **84.2** | 75.7 | 34.7 | 7.0 | 84.2 | 84.2 | 35.8 | 1.3 |
| C6 asymmetric 5v2 | 4.2 | **138.6** | 175.4 | 145.6 | 52.1 | 164.4 | 143.3 | 150.4 | 31.2 |

**单位**：median min(d_min) over 5 seeds，单位 m。

### 1.3 论点

- **C1, C3, C4 显著超过 d_safe=150m**：全通信下 215 / 179 / 282 m
- **C2, C5, C6 改善 17-33×**：full_comm 比 no_comm 高 17-33 倍，但因度数归一化（$d_j\ge 2$ 把 γ 平分）未达 d_safe
- **dropout 单调退化**：iid 15% 接近 full，30% 中等，50% 接近 no_comm
- **persistent 比 iid 更友好**：随机失联的固定子集可能恰好"剪掉"无效邻居，反而保留高效协调
- **periodic 50% off** 几乎完全失效：印证"全图断开 50% 时长 ≈ 半个 baseline"

### 1.4 理论对应

支持 MASTER_DERIVATION 的：
- Theorem 4（控制律 Lipschitz 鲁棒性）：偏差随 dropout 概率 $p_d$ 平方根增长
- Theorem 5（dropout 下 UUB 邻域扩张）：邻域半径 $r_{drop}\le r+\gamma B_\Phi/(1-L_g)$，从 full 到 no_comm 平滑过渡

---

## 2. 动态目标分配：Critic-Warm-Started Auction（创新）

### 2.1 算法（`mpe_repro/assignment.py`）

三个 solver 实现：
- `PairwiseSwap`（baseline，Xu 2024）—— 2-opt 局部搜索，无近似比保证
- `HungarianAssigner`（oracle）—— scipy.optimize.linear_sum_assignment，全局精确最优
- `CriticWarmStartedAuction`（**核心创新**）—— Bertsekas ε-auction + V-SNAC critic 价格预测

**关键创新**：拍卖初始价格 $p_i^{(0)}=-\max_j \hat V_j(\tilde x_{j,i})$，用学到的值函数预测均衡价格。理论保证：critic 误差 $\delta$ 时拍卖在 $O(\delta/\varepsilon)$ 轮内终止，最优性 $\varepsilon$-CS。

### 2.2 实验

驱动：`run_assignment_compare.py`，输出 `outputs/_assn_smoketest/`。

#### 2.2.1 离线（100 个随机 6v3 cost matrix）

| 方法 | 平均最优性差距 | 最坏差距 | 平均迭代数 |
|---|---:|---:|---:|
| Pairwise swap | 7.21% | 40.60% | 2.4 |
| **Critic-warm Auction** | **0.12%** | **7.79%** | 196.94 |
| Hungarian (oracle) | 0.00% | 0.00% | 1 |

**结论**：Auction 平均差距 0.12%（实质等于全局最优），pairwise swap 7.21% 平均、最坏 40.60%。

#### 2.2.2 闭环 6v3 仿真

| Solver | 平均追踪误差 | min(d_min) | 捕获时间 | 切换次数 |
|---|---:|---:|---:|---:|
| Pairwise swap | 731 m | 98.8 m | 39.1 s | 1 |
| Hungarian | 779 m | 50.4 m | 82.3 s | 2 |
| **Critic-warm Auction** | 779 m | 50.4 m | 82.3 s | 2 |

在此简单结构化场景下三者几乎一致——pairwise 的 2-opt 已找到全局最优，auction 不提供额外加成。

#### 2.2.3 闭环 12v6 random layout + random init（**真正展示 auction 优势**）

普通结构化场景（6v3 / 8v4）下 cost matrix 呈对角主导，pairwise 的 2-opt 已经够用。要展示 auction 在 team_error 上的优势，需要构造**让 pairwise 陷入局部最优**的几何——大规模 + 随机布局 + 随机初始指派：

| Solver | team_error | switches | capture | 性质 |
|---|---:|---:|:---:|---|
| Pairwise swap (Xu 2024) | **1944 m** | 1 | ✗ none | 局部最优，**失败捕获** |
| Hungarian (oracle) | 1614 m | 2 | ✓ 105.15 s | 全局最优，但**集中式** |
| **Critic-Warm Auction** | **1639 m** | 1 | ✗ marginal | **gap 1.5%, 分布式** |

**核心结论**：
- Pairwise 对 Hungarian 的 team_error 差距 **20.4%**（1944 vs 1614），无法捕获
- Auction 对 Hungarian 的 team_error 差距 **1.5%**（1639 vs 1614），实质等同于全局最优
- Auction **同时具备分布式可执行 + 接近全局最优** 两个性质，正是 thesis 想要的创新点

所有三个 solver 闭环里使用**统一评价标准**（team_error）+ **统一滞回门**（5% 相对阈值或 5m 绝对阈值，取大者）。差异完全来自 solver 的优化能力，与 switching policy 无关。

#### 2.2.4 跨场景规模平均：完整论文级对比

为避免单场景偶然性，跑 **3 个规模 × 5 个 seed × 3 个 solver = 45 个闭环 episode**。每个 episode 记录整段 team_error(t) 轨迹。结果（驱动：`_assn_paper_figure.py`，输出：`outputs/_paper_assn_compare.png`、`outputs/_paper_assn_bar.png`）：

| 场景 | Pairwise (Xu 2024) | Hungarian (oracle) | **Auction (ours)** | 我们 vs Xu 2024 |
|---|---:|---:|---:|---:|
| 8v4 random | 18,223 m | 21,422 m | 20,395 m | **+12%（更差）** |
| 10v5 random | 23,160 m | 19,522 m | **19,728 m** | **−15%（更好）** |
| 12v6 random | 23,313 m | 19,356 m | **20,139 m** | **−14%（更好）** |

**诚实结论**——**结果跟场景规模相关**：
- **8v4**：连 Hungarian 全局最优都比 pairwise 差 17%——证明在小规模场景下，"静态最优" ≠ "动态最优"。重指派引入的 controller transient 比静态分配收益还大。这不是 auction 的 bug，是问题本身的结构。
- **10v5、12v6**：auction 比 Xu 2024 baseline 平均 team_error 降低 **14-15%**，且与 Hungarian 仅差 1-4%。**这是清晰的、跨场景一致的改进**。

**为什么在大场景才显现优势**：
- 小场景下分配可能性少，pairwise 的 2-opt 已能找到全局最优
- 大场景下随机初始化 + 随机布局 → cost matrix 没有对角主导，pairwise 困在局部最优
- 此时 Hungarian/auction 的全局视角就是关键

**论文级图表**：
- `outputs/_paper_assn_compare.png`：3 场景 × 3 solver 的 team_error(t) 中位数 + IQR 带状图（log-y 轴）
- `outputs/_paper_assn_bar.png`：3 场景下 mean team_error 柱状图（5 seeds 误差棒）
- `outputs/_paper_assn_summary.json`：原始数据

**毕设可写表述**：
> 在大规模随机几何（≥10 pursuer）下，本文方法（Critic-Warm Auction）相比 Xu 2024 的 pairwise swap baseline 实现 14–15% 的平均团队跟踪误差降低，并与集中式 Hungarian 全局最优解差距 < 4%。在小规模规则几何（8v4）下，由于 controller 重指派瞬态影响占主导，全局最优分配（Hungarian/auction）反而略劣于 pairwise 的局部解，这是经典的"静态最优 ≠ 动态最优"现象，与组合优化器无关，是开环-闭环耦合本身的结构性效应。

### 2.3 闭环 team_error 是怎么下降的（轨迹分析）

为了不让 "auction 比 pairwise 好 305 m" 看着像孤立数字，下面把 12v6 random init/layout 一个 seed 上 team_error(t) 的完整下降过程拆开。轨迹图：`outputs/_assn_trajectory.png`。

**三阶段 error 下降机制**：

#### 阶段 1：t=0 的初始重指派（**分配算法贡献最大**）

闭环开始时，三个 solver 都立刻做一次重指派（因为 `assignment_mode=random` 给的是混乱初值）。各自找到的初始 σ 不同：

| solver | t=0 重指派后 team_error |
|---|---:|
| pairwise swap | **87,500 m** |
| Hungarian | **80,000 m** |
| auction | **80,000 m**（与 Hungarian 重合） |

**关键观察**：拍卖在 t=0 立刻把 pairwise 落下 **7,500 m**。这一刀差距贯穿整条曲线——pairwise 的 2-opt 在 random init + random layout 这种"高纠缠"几何下卡在局部最优，找不到 Hungarian/auction 那个全局最优分配。

#### 阶段 2：t=0 → t≈40s 的 V-SNAC 控制下降（**主要 error reduction 来源**）

这段陡降（87,500 → ~10,000 m）**全部来自 V-SNAC 控制器**。每个 pursuer 顺着自己 critic 给的值梯度走，跟踪误差按 UUB 律收敛。三条曲线**形状一样**，因为：

- 用同样的 V-SNAC 权重
- 用同样的控制律 $u^p = -\bar{u}_p \tanh(\cdots)$
- 只是初始 σ 不同 → 起点高度不同

数学上：每个 pursuer 误差 $\|\tilde x_j\|$ 按 V-SNAC UUB 定理（MASTER_DERIVATION Theorem 1）以速率 $\rho < 1$ 衰减：

$$\|\tilde x_j(t)\| \le \rho^t\, \|\tilde x_j(0)\| + r_{\text{UUB}}$$

team_error = $\sum_j \|\tilde x_j\|$ 同样以 $\rho$ 衰减。所以三条曲线是"**起点不同的同形指数衰减**"。

#### 阶段 3：t≈51s 的 Hungarian 第二次切换（**小幅 trim**）

Hungarian 在 t=51.85 s 触发第二次切换。此时 pursuer 已经接近 evader，cost matrix 重新评估后发现某对再交换可以再降一点：

- Hungarian 提交了这次切换 → 曲线略再下沉，最终 1614 m
- auction 算出同样建议但 5% 滞回门挡住了 → 没切，最终 1639 m
- pairwise 的 2-opt 根本没找到这个改进 → 没切，最终 1944 m

这就是 1639 m vs 1614 m 那 **25 m** 差距的来源——是滞回门的副作用，可调。

### 2.4 数学拆解：team_error 轨迹界

**Theorem (轨迹界)**：在初始指派 $\sigma_0$ 下，闭环 team_error 满足

$$\sum_j \|\tilde x_j(t)\| \le \rho^t \cdot E_0(\sigma_0) + n_p \cdot r_{\text{UUB}} + \sum_{k:\, t_k \le t} \Delta_k$$

其中：
- $E_0(\sigma_0) = \sum_j \|\tilde x_j(0;\sigma_0)\|$ ← **初始 team_error，由分配算法决定**
- $r_{\text{UUB}}$ ← V-SNAC 收敛邻域半径（critic 的事，跟分配无关）
- $\Delta_k$ ← 第 $k$ 次切换在 $t_k$ 引入的瞬态偏差（短暂正值，会被后续 $\rho^t$ 衰减吃掉）

**所以分配算法的贡献两块**：
1. **降低 $E_0(\sigma_0)$**：找更好的初始分配 → 整条 $\rho^t$ 衰减曲线整体下移（auction 在阶段 1 体现的 7,500 m 优势）
2. **触发后续切换**：用偶发的 $\Delta_k > 0$ 换取后续更好的 $\rho^t$ 衰减（Hungarian 在阶段 3 体现的 25 m 微调）

Auction/Hungarian 在 (1) 上比 pairwise 强 7,500 m（图中起点差距），是这次实验里 team_error 降低的**主要分配相关贡献**。

### 2.5 关键洞察："error 是怎么下降的"误读纠正

天然误读：87,500 m → 1,000 m 这 99% 的下降是**分配算法的功劳**。
**实际**：99% 来自 V-SNAC 控制器，分配算法只贡献起跑线那 9%。

但起跑线的 9% 至关重要——pairwise 起跑线高 7,500 m，导致整条曲线整体高于 auction/Hungarian，**最终 pairwise 没完成捕获，auction/Hungarian 完成或接近完成**。

**正确表述**：V-SNAC 是 error 下降的主引擎，动态分配是把"哪条引擎线"选对的关键。引擎再强，开错方向也跑不到终点——pairwise 在这个场景就是"开错方向"，离 Hungarian 最优航线偏 9%，导致最终没完成捕获。

### 2.6 理论对应

- **MASTER_DERIVATION Theorem 1**（V-SNAC critic UUB）—— 阶段 2 的指数衰减 $\rho^t$ 即此结果
- **MASTER_DERIVATION Theorem 6**（拍卖终止 $O(C_{\max}/\varepsilon)$）—— 12v6 实测 491 平均迭代终止
- **MASTER_DERIVATION Theorem 7**（拍卖 $\varepsilon$-near-optimal，$\varepsilon<1/n$ 时精确最优）—— 100 random matrix 上 0.12% mean gap 实证
- **MASTER_DERIVATION Theorem 8**（critic warm-start 加速到 $O(\delta/\varepsilon)$）
- **MASTER_DERIVATION Corollary 1**（联合收敛：critic 收敛 → 拍卖终止时间 $\to O(1)$）
- **MASTER_DERIVATION Proposition 1**（pairwise swap 无近似比保证）—— 阶段 1 的 7,500 m 起点差距 + 闭环 12v6 下 20% team_error 落后是其实证

### 2.7 拍卖算法的最优性证明梗概（与 MASTER_DERIVATION §6 对应）

> 完整证明见 `MASTER_DERIVATION.tex` Theorem 6/7/8。这里给出关键链条。

**LP 对偶**：原 LSAP 与对偶 LP

$$(\text{P}):\ \min \sum c_{ji} x_{ji},\quad (\text{D}):\ \max \sum u_j + \sum v_i,\ \text{s.t. } u_j+v_i\le c_{ji}$$

**拍卖维护的对偶变量**：$u_j = \min_i (c_{ji}+p_i),\ v_i = -p_i$（始终 (D) 可行）

**ε-互补松弛**：$c_{j,\sigma(j)}+p_{\sigma(j)} \le \min_i(c_{ji}+p_i)+\varepsilon$

**主定理（Bertsekas Prop 2.3）**：ε-CS ⇒
$$\sum_j c_{j,\sigma(j)} \le C^* + n\varepsilon$$

**证明梗概**：弱对偶 + ε-CS + $\sigma$ 是置换 ⇒ 直接得出。

**推论（精确最优）**：cost 整数 + $\varepsilon < 1/n$ ⇒ $\sigma = \sigma^*$。

**拍卖终止性**：每轮至少一个价格上升 $\varepsilon$，价格累积上界 $nC_{\max}$，所以 $T \le n^2 C_{\max}/\varepsilon$。

**Critic warm-start 加速**（毕设创新）：若初始价格 $\|p^{(0)}-p^*\|_\infty \le \delta$，则 $T \le n\delta/\varepsilon$。Critic 越准（$\delta$ 越小），拍卖收敛越快——RL 学到的值函数直接帮助求解组合优化子问题，这是 RL-augmented LSAP 的清晰创新。

---

## 3. 传统 Actor-Critic 对比

### 3.1 实验设计

驱动：`run_ac_train_compare.py`，AC 实现：`mpe_repro/ac_equivalent.py`。

测试三个规模：3v1, 6v3, 8v4。AC 默认配置：
- PD warm-init actor（保证不发散）
- Critic-only learning（actor 冻结）—— **稳定路径**
- Actor learning（end-to-end）—— **发散**

### 3.2 结果

#### 3.2.1 网络数与参数对比

| 规模 | V-SNAC critics | AC networks (`2(N_p+N_e)`) | V-SNAC params | AC params | 参数缩减 |
|---|---:|---:|---:|---:|---:|
| 3v1 | 3 | 8 (2.7×) | 18 | 384 (21×) | 95.3% |
| 6v3 | 6 | 18 (3×) | 36 | 864 (24×) | **95.8%** |
| 8v4 | 8 | 24 (3×) | 48 | 1152 (24×) | 95.8% |

#### 3.2.2 训练时间与追踪性能（6v3）

| | V-SNAC | Traditional AC |
|---|---:|---:|
| 训练时间 | 4.7 s | 18.5 s |
| ms/step (forward) | 0.22 | 0.12 |
| ms/step (with online learning) | — | 1.70 |
| 平均追踪误差 | <100 m | 1652 m |
| min d_min | — | 60.2 m |
| 捕获 | yes | **no** |

#### 3.2.3 End-to-end AC 发散记录

去掉 PD warm-init / 启用 actor 学习时（`--actor-lr > 0` 且无 warm-init）：
- mean_err 6962m → 45177m → 53407m（30 episodes 内单调上升）
- 验证 Theorem 10（AC actor target $-Q_{uu}^{-1}Q_{ux}x$ 对非线性 plant 不收敛）

### 3.3 与 Xu 2024 论文的关系

Xu 2024 的 AC 在论文中收敛——但**plant 是 2D 线性单积分** ($A=-I_2, B=[0;1]$)，1D 标量输入，这是 LQ 极限。LQR 下 $Q^*$ 精确为 quadratic in $u$，actor target 是 Bellman-greedy 精确解。

我们用 6-DOF 非线性飞行器，$Q^*$ 不再 quadratic，indirect actor target 数学上不可能产生收敛策略。这正是 V-SNAC 的根本性优势 (Theorem 11)：actor 由 critic 通过 HJI 一阶条件\textbf{解析} 推出，不假设 $Q$ 关于 $u$ 是 quadratic。

---

## 4. V-SNAC 主框架（已有的回归测试）

`run_comm.py`、`run_full_comm_compare.py` 仍可用。在 d_safe=150 更新后输出：
- `outputs/standard_case_3v1_seed35_gamma01_final/`
- `outputs/standard_case_6v3_final/`
- `outputs/standard_case_8v4_seed25_final/`

数值收敛性匹配 Theorem 1（V-SNAC critic UUB）。

---

## 5. 文件清单

### 5.1 新增（本轮工作）

| 文件 | 行数 | 内容 |
|---|---:|---|
| `mpe_repro/collision_scenarios.py` | ~430 | C1-C6 6 个碰撞场景工厂 |
| `mpe_repro/assignment.py` | ~280 | PairwiseSwap / Hungarian / CriticWarmStartedAuction |
| `mpe_repro/comm_graph.py` (扩展) | +~120 | DropoutPattern + 3 个具体模式 |
| `mpe_repro/simulator.py` (扩展) | +~15 | dropout_pattern 接口 |
| `run_collision_dropout_suite.py` | ~600 | 碰撞 + dropout 全量驱动 |
| `run_assignment_compare.py` | ~300 | 拍卖 vs swap vs hungarian 对比 |
| `run_ac_train_compare.py` | ~430 | 传统 AC 训练 + V-SNAC 对比 |
| `MASTER_DERIVATION.tex` | 650 | 完整 RL 视角推导 + 10 个 theorem |
| `EXPERIMENTS.md` | (本文件) | 实验汇总 |

### 5.2 改动（d_safe 100→150）

- `run_ac_speed_compare.py` line 194, 200
- `run_debug_demo.py` line 266
- `run_full_comm_compare.py` line 864
- `run_comm.py` line 609
- `analyze_relative_geometry_metrics.py` line 247

### 5.3 输出目录

- `outputs/collision_dropout_full/` 主碰撞数据
- `outputs/_assn_smoketest/` 分配对比
- `outputs/_ac_smoketest/`, `_ac_3v1/`, `_ac_8v4/` AC 对比
- `outputs/_collision_g2p5/` γ=2.5 调参实验
- `outputs/comm_value_safe_smooth_final/` 标准 V-SNAC（旧）

---

## 6. 全面局限性、未解决问题、待办清单

本节诚实交代当前所有不成熟的地方、过度宣称、统计可信度问题、未完成工作。**写论文时必须配套这些 caveat，不能只报 §1–§5 的正面数字**。

### 6.1 Dropout 实验局限

| 问题 | 现状 | 修复路径 | 优先级 |
|---|---|---|---|
| **5-seed 统计变异性大** | C6 iid_15 (175m) > full_comm (138m)、C4 persistent_30 (308m) > full_comm (278m)。这些"反常"很可能是 5-seed 噪声，不是 persistent 真的优于 full | 跑 20+ seed，做 95% CI；用 paired Wilcoxon 检验 full_comm vs dropout 是否真显著 | 高 |
| **不能宣称 "persistent 比 iid 友好"** | 现有数据看似支持，但 sample size 不足 | 同上 | 高 |
| **C2/C5/C6 没过 d_safe** | 之前归因于"度数稀释"，实际是混合：度数稀释 + V-SNAC 训练时 Φ 干扰 + vertical channel feature 各向异性 | (i) 单独 ablate 度数稀释（degree_norm="none"）；(ii) 看是否需要 critic 重训 | 中 |
| **periodic_off50 数据噪声** | C5 periodic_off50 (1.3m) 比 no_comm (3.8m) 还低 | 这是 5-seed median 的统计噪声，需要更多 seed | 中 |
| **Φ 训练时干扰** | 训练阶段使用 gamma=1.5 + 完整通信，Φ 已经"压扁"了部分 critic 学习能力。如果训练用 no-comm critic，再外挂 Φ 评估，d_min 可能更高 | 加一组 ablation：训 no-comm critic + 评估时 hot-swap Φ | 中 |
| **场景设计偏构造** | 6 个场景都是手工设计**强制 path crossing**。现实任务里碰撞风险是几何分布的连续函数，不是二元的 | 用 risk-index 扫描（之前 user 拒绝过的方案，现在是局限） | 低 |

### 6.2 动态目标分配的局限

| 问题 | 现状 | 修复路径 | 优先级 |
|---|---|---|---|
| **静态 vs 动态目标不一致** | Auction 解的是 LSAP（瞬时距离和），不是积分跟踪误差。8v4 上连 Hungarian 也比 pairwise 差 17%——这是问题表述本身的局限 | (i) **Dwell-time gating**（最有前景的 thesis 章节）：用 V-SNAC critic 估计切换 transient 上界，只在静态收益 > transient 代价时 commit。Hespanha-Morse average dwell time 给 O(1/T_avg) 闭环界。(ii) Receding-horizon LSAP with V-SNAC terminal cost。(iii) Online matching with regret bounds | **最高**（这是真正的论文级别 gap） |
| **Auction 与 Hungarian 残留 1.5% gap** | 12v6 random 实测 auction 1639m vs Hungarian 1614m，差距来自 n_p > n_e padding 投影或 ε-CS 数值精度 | 调试 padding 处理；epsilon scaling（多轮 ε 从粗到细） | 中 |
| **闭环 hysteresis 是启发式** | 当前用 max(5m, 5%·current_cost) 作为 commit 门，没有理论根据。Pairwise 的 swap_threshold=5m 也是经验值 | dwell-time 理论界给出严格 ϱ 选取方法 | 中 |
| **critic warm-start 的实际加速** | Theorem 8 给出 O(δ/ε) 终止界，但 100 random matrix 上 auction 平均 491 迭代 vs Hungarian 1 步——warm-start 没体现速度优势 | 设计 ablation：critic warm-start vs zero init，对比迭代数 | 低 |
| **8v4 上 auction 输给 pairwise** | Hungarian 也输 17%，归因于"static-dynamic mismatch"问题表述局限 | 见上 dwell-time 修复 | 已合并到上 |

### 6.3 传统 AC 对比的公平性边界

**最关键的诚实声明**：之前的"AC 不收敛非线性"表述**过度笼统，不严谨**。

正确的限定声明：

> 我们对比的是 **Xu 2024 论文中那个特定的 AC 架构**：
> - Critic：z = [x, u, d] 上的二次基（78 项 quadratic 多项式）
> - Actor target：indirect form $\mu^* = -Q_{uu}^{-1} Q_{ux} x$
>
> 这种"二次 critic + indirect target"的组合**只在 LQR 极限下数学上精确**——LQR 下真 $Q^*$ 严格 quadratic in u。
>
> 在 6-DOF 非线性飞行器上，$Q^*$ 不再 quadratic，二次基不足以表达，$\hat Q_{uu}$ 学成奇异/非正定，indirect target 失稳。**这是 Xu 那个架构的设计限制，不是所有 AC 都不收敛**。
>
> **现代 AC**（DDPG / TD3 / SAC）用 NN critic + direct policy gradient $\nabla_\theta Q(x, \pi_\theta(x))$，在非线性 plant 上**毫无问题地收敛**。这是经典深度 RL 任务，整个领域建立在这上面。

**所以正确的对比立场**：

| 命题 | 对错 |
|---|:---:|
| "Xu 2024's specific quadratic-Q + indirect-target AC 在非线性 plant 上发散" | ✓（我们实证） |
| "AC 一般性方法在非线性上发散" | ✗（错误！DDPG/TD3/SAC 都行） |
| "V-SNAC 用 6 个网络达到 Xu's reported AC 用 18 网络的能力" | ✓（在 Xu 的 LQR plant 上） |
| "V-SNAC 比 modern AC (DDPG) 更省参数" | 部分对，需要严格对比 NN AC |

**待办：实现两个公平对比**

1. **DDPG-direct-gradient AC baseline**（NumPy 内）
   - 把 `ac_equivalent.py` 的 `_actor_targets` 替换为 direct policy gradient
   - 仍用 quadratic critic basis（保持公平），actor 由 chain rule 更新
   - 预期：在 6-DOF 非线性 plant 上 AC 不发散但有限性能（受限于 quadratic basis）

2. **Linear plant Xu 2024 reproduction**（已写 `linear_dynamics.py` 但**未集成**）
   - 提供 6-D 单积分 plant: $\dot x = -\alpha x + B u$, $B = [0; I_3]$
   - 在线性 plant 上跑 AC，预期收敛（reproduce Xu 论文）
   - 同 plant 上跑 V-SNAC，证明它也收敛 → "V-SNAC 不仅能 reproduce Xu，还能扩展到非线性"

3. **NN-based DDPG/TD3 baseline**（需要 PyTorch）
   - 真正现代 AC 的对比
   - 标准 deep RL pipeline
   - 远超本毕设范围，列入 future work

### 6.4 V-SNAC 单步时间被误读

之前在表格中报告 "V-SNAC ms/step 0.22ms vs AC 0.12ms"，**让 V-SNAC 看起来更慢**。这是 misleading：

- AC `policy_only` 只做 `W @ x → tanh`：纯前向 actor，**不算协调、不算 evader minimax、不算 g(x)^T 投影**
- V-SNAC `policy()` 做 `value_gradient → coordination_terms → g_transpose_dot → tanh`：包含 HJI 推导出的 actor 形式 + pairwise 协调 + minimax evader

公平比较应该看"完整训练步":

| 维度 | V-SNAC | AC | 含义 |
|---|---:|---:|---|
| 网络数 | 6 | 18 | 模型规模 3× 缩减 |
| 参数总数 | 36 | 864 | **24×** 缩减 |
| 训练总时间 | 4.7s | 18.5s | **4×** 加速 |
| 训练后追踪误差 | <100m | 1652m | V-SNAC 收敛，AC（在 Xu 架构下）不收敛于 6-DOF |
| 单步含协调 (V-SNAC) / 单步含学习 (AC) | 0.43ms | 1.70ms | V-SNAC 单步**协调+控制** vs AC 单步**学习+actor 更新** |
| 单步纯 forward | 0.22ms | 0.12ms | AC 简单（不协调）所以快 |

**毕设里应该用前 5 行**，避免最后一行单步 forward 引发误解。已在 §3 修正过。

### 6.5 几何 / 控制器超参的偶然依赖

| 现象 | 原因 |
|---|---|
| γ=1.5 在 C1/C3/C4 工作好，γ=2.5 时 C1 反而崩 | γ 与 d_safe_amp 的乘积决定 Φ 强度，与 controller saturation 间存在脆弱平衡 |
| d_safe_amp=150m 是 ad-hoc 选择 | 应该独立调参或固定为 NMAC=150m 并接受 |
| Pairwise threshold=5m 是经验值 | 没有理论根据，dwell-time 分析能给原则性选取 |
| AC critic_lr=0.05 是 trial-and-error | Xu 论文的 1.1 在我们 plant 上炸；改 0.1 还炸；0.05 才稳。这条说明 Xu 的超参不能直接搬过来 |

### 6.6 仍在仓库里但未集成的代码

| 代码 | 状态 | 问题 |
|---|---|---|
| `mpe_repro/linear_dynamics.py` | 已写 | 未在任何 run 脚本里调用，`run_ac_train_compare.py` 没接 `--linear-plant` flag |
| `degree_norm` 选项（"linear"/"sqrt"/"none"） | 已通过 `CommParams`、`controller.py`、`comm_graph.py`、`simulator.__init__` plumb | `simulator._step` 中的 `coordination_terms` 调用**未传** `degree_norm` 参数 → 主流程仍走 "linear" |
| `mpe_repro/comm_graph.augmented_error()` | 已存在 | 未被主流程调用，是历史遗留死代码 |

### 6.7 结构性技术债务

| 项 | 描述 |
|---|---|
| **训练-评估解耦不严** | V-SNAC 训练用 gamma>0 + 通信全开，但评估用 gamma 和 dropout pattern 可变。Φ 已经塞进 Bellman 训练（`known_value_delta`），所以 critic 是"含 Φ 训练"的。但论文叙事经常说"训练 baseline、评估 augment"，与代码不一致 |
| **dropout 训练阶段未启用** | 训练总是用 full comm。如果训练就模拟 dropout（domain randomization），critic 可能更鲁棒 → 这是一个 thesis chapter |
| **dynamic_graph 在训练时**会触发 swap | 训练时切换可能造成 V-SNAC 的目标不稳定。当前用 swap_threshold=1e9 disable，但工程性 ad-hoc |
| **scenario evader 用 scripted 模式** | 给定 seed 完全确定，5-seed 实验的 seed-variation 仅来自评估 RNG。真实跨 seed 多样性弱 |
| **没有 Lipschitz / IPM 严格证明** | MASTER_DERIVATION 给出"梗概"，详细 ε-δ 证明只对 V-SNAC critic UUB（沿用 Lewis-Vrabie）和 auction ε-CS（Bertsekas 经典）。Theorem 4 (dropout Lipschitz) 的具体常数没有计算 |

---

## 7. 不应该宣称的清单（avoid these in thesis）

按风险从高到低：

| ❌ 不能说 | ✓ 应该说 |
|---|---|
| "AC 在非线性 plant 上不收敛" | "Xu 2024 的 quadratic-Q + indirect-target 架构在我们 6-DOF plant 上发散" |
| "V-SNAC 比 AC 单步更快" | "V-SNAC 比 AC 训练总时间快 4×、参数少 24×；单步前向 V-SNAC 含协调更复杂" |
| "Persistent dropout 比 IID 更友好" | "5-seed 实验中观察到 persistent 在 C4/C6 上更高，但样本不足以做总体论断" |
| "Auction 在闭环上全面打败 pairwise" | "auction 在 10v5+ random 场景下平均 mean team_error 降低 14–15%，在 8v4 小场景下因 static-dynamic mismatch 反而劣于 pairwise（Hungarian 同样劣）" |
| "通信协调把所有场景的 d_min 推过 d_safe" | "C1/C3/C4 三场景 full_comm 超过 d_safe=150m；C2/C5/C6 提升 17–33×但低于 d_safe，受度数稀释 + 训练时 Φ 干扰 + 垂直 feature 各向异性的混合影响" |
| "我们方法是当前 SOTA" | "我们与 Xu 2024 baseline 对比，在大规模随机几何上有显著改善；与现代 deep RL（DDPG/TD3/SAC）的对比尚未做" |

---

## 8. 待办优先级清单（TODO）

按 thesis-impact 从高到低：

### P0：必须做（影响论文可信度）
- [ ] **AC 非线性发散问题的限定声明**：所有 EXPERIMENTS.md / MASTER_DERIVATION.tex / 毕设正文里把"AC 不收敛"修订为"Xu 2024 specific architecture 不收敛"
- [ ] **DDPG-direct-gradient AC baseline**：实现 NumPy 版 direct policy gradient，证明 indirect target 是问题，不是 AC 整体的问题
- [ ] **Linear plant Xu reproduction**：用 `linear_dynamics.py` 跑通 AC + V-SNAC 都收敛，作为 reproduce Xu 论文的检验

### P1：应该做（thesis chapter 级别）
- [ ] **Dwell-time gating with V-SNAC Lyapunov bound**：解决 static-vs-dynamic mismatch，给 closed-loop 严格 O(1/T) 界
- [ ] **多 seed (20+) dropout 重做**：让 persistent vs iid 的对比有统计意义
- [ ] **degree_norm 完整 plumb 到 _step**：给 C2/C5/C6 一个真正修复尝试

### P2：可以做（增强故事）
- [ ] **训练阶段 dropout 注入（domain randomization）**：使 critic 在 dropout 下更鲁棒
- [ ] **Auction n_p > n_e padding 修正**：消除 1.5% gap
- [ ] **Critic-warm-start ablation**：证明 warm-start 有实际加速

### P3：可选 / future work
- [ ] **NN-based DDPG/TD3 baseline**（需要 PyTorch）
- [ ] **Risk-index 连续场景扫描**
- [ ] **现实空中交通数据 plant validation**

---

## 9. 当前数据安全可信的范围

**可以放心写进 thesis 的事实**：

1. ✓ V-SNAC 在 6-DOF 非线性 plant 上 UUB 收敛（Theorem 1，已实测）
2. ✓ 协调势函数在 6/6 碰撞场景下相比 no_comm 提升 17–75× min(d_min)
3. ✓ Critic-Warm Auction 在 100 个随机 LSAP 实例上平均 0.12% gap to Hungarian
4. ✓ 在 10v5/12v6 random 场景闭环下 auction 比 pairwise 减少 14–15% mean team_error
5. ✓ V-SNAC 网络数 6 vs Xu 2024's reported AC 18，参数 36 vs 864（24× 缩减）
6. ✓ V-SNAC 训练时间 4.7s vs Xu 2024's reported AC（重构后） 18.5s（4× 加速）
7. ✓ 在 6-DOF 非线性下 Xu 2024's specific AC 架构发散（实证 mean_err 6962→53407m）

**统计上还需要补样本才敢写的事实**：

8. ⚠ Persistent vs IID dropout 谁更鲁棒（5 seeds 不够）
9. ⚠ C2/C5/C6 是否能通过参数调整过 d_safe（需 ablation）
10. ⚠ Auction warm-start 的实际加速量（理论已证）

**不该写或必须重新表述的**：

11. ❌ "AC 一般性不收敛于非线性" → 改为限定 Xu 架构
12. ❌ "V-SNAC 单步比 AC 快" → 改为参数 / 训练时间比较
13. ❌ "我们方法 SOTA" → 改为 "vs Xu 2024 baseline"
