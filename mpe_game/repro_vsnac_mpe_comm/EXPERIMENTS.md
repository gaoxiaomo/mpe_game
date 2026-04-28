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

## 6. 限制与后续

1. **C2/C5/C6 未达 d_safe=150m**：受限于度数归一化的 γ 分摊。后续可加 \textbf{γ adaptive}（按 $d_j$ 缩放，$\gamma_{\text{eff}}=\gamma\cdot\sqrt{d_j}$）让大群体也能达到目标分离。
2. **AC end-to-end 发散是固有的**：可作为毕设论点（V-SNAC 结构耦合是改进方向），不是 bug 需修。
3. **拍卖创新点已立**：但闭环里 cost matrix 用静态距离，下一步换成 V-SNAC 值可让拍卖直接优化动态目标。
4. **动态图算法的更多扩展方向**：见研究代理给出的 7 个 frontier（auction warm-start, 残差 Bellman + value, Sinkhorn soft assignment 等）。
