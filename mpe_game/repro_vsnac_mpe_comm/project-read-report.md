# Project Read Report: repro_vsnac_mpe_comm

Generated at: 2026-04-05

## 1. Project Snapshot
- Goal: 复现 Xu 2024 的多追逐者多逃逸者非线性追逃博弈/V-SNAC 方法，并在此基础上加入“同组追逐者通信协调”层，重点提升空间分离与碰撞规避能力。
- Stack: Python + NumPy + Matplotlib，本地脚本式实验工程，没有看到 `pyproject.toml` / `requirements.txt` 这类统一依赖入口；主入口是直接运行脚本。
- Entrypoints:
  - `run_comm.py:229` 负责批量训练、三种通信模式评估、出图和报告导出。
  - `run_collision_demo.py:30` 负责受控碰撞场景，突出通信协调对 `d_min` 的改善。
  - `run_cross_demo.py:49` 负责更简化的交叉路径演示，帮助解释机制。
- Config surface:
  - `mpe_repro/config.py:9` 定义飞行器、学习、控制、特征、场景、通信参数。
  - `mpe_repro/general_scenarios.py:15` 定义通用场景生成器。
  - `0331Report.md:5` 和 `PAPER_COMM_VSNAC.tex:33` 解释了作者自己的研究动机和结果组织方式。

## 2. 这项目到底在做什么
这个项目的“论文复现层”主要对应 Xu 2024 的三件事：非线性 6-DOF 追逃动力学、基于 HJI 的饱和控制律、以及 V-SNAC + 动态目标交换图的求解框架。原论文第一页摘要和前五页的算法部分已经明确了这三个组件：多智能体追逃、动态目标图、V-SNAC 单网络 critic，以及输入饱和下的解析控制律（原论文 PDF 第 1-5 页；另见 `mpe_repro/dynamics.py:8`, `mpe_repro/features.py:8`, `mpe_repro/graph_switch.py:6`, `mpe_repro/controller.py:79`）。

项目的“作者扩展层”则非常明确：不再只让追逐者各自追目标，而是在“追同一个逃逸者”的追逐者之间建立通信图，并把一个解析协调项加到控制律里，让追踪和协调共享同一个 `tanh` 饱和控制通道。这个思路在周报里讲得最清楚：原论文没有直接约束同组追逐者之间的相对距离，因此会出现路径交叉甚至碰撞；于是作者加入通信协调势函数来改善安全性与空间分离（`0331Report.md:7`, `0331Report.md:9`）。

## 3. Architecture Narrative
运行主实验时，`run_comm.py` 先根据命令行参数收集 case，再为每个 case 构造 `GeneralScenarioSpec`，生成 pursuer/evader 初始状态、编队偏移和初始指派（`run_comm.py:470`, `run_comm.py:613`, `mpe_repro/general_scenarios.py:210`）。随后它创建 `MPECommSimulator`，把动力学、特征映射、控制器、通信图和场景参数装配起来（`run_comm.py:246`, `mpe_repro/simulator.py:71`）。

训练阶段走 `MPECommSimulator.train_policy()`：初始化 critic 权重，滚动采样轨迹，把 Bellman 差分样本塞进 `ReplayLeastSquares`，然后用带 ridge/Tikhonov 先验的最小二乘去更新权重（`mpe_repro/simulator.py:420`, `mpe_repro/offpolicy_ls.py:16`）。评估阶段走 `evaluate_policy()`：在同一组已训练权重上分别切换 `full_comm`、`no_comm`、`dropout` 三种通信模式，记录轨迹、误差、控制输入、`d_min`、formation error 和目标切换（`run_comm.py:264`, `mpe_repro/simulator.py:508`）。

真正的控制律在 `CommVSNACController.policy()`：先按原论文的 tracking error 算 value gradient，再把 formation gradient 作为额外项加进去，最后一起送进 `tanh` 饱和控制律（`mpe_repro/controller.py:144`, `mpe_repro/controller.py:175`）。这意味着当前实现更接近“增广值函数/势函数梯度注入”，而不是“直接把通信项写进 critic 训练目标”。

## 4. Runtime Flows
- Flow A: 批量主实验
  - `run_comm.py:576` 解析参数
  - `run_comm.py:619` 组装 jobs
  - `run_comm.py:637` 多进程并行跑 case
  - `run_comm.py:229` 单 case 训练/评估
  - `mpe_repro/simulator.py:420` 训练 critic
  - `mpe_repro/simulator.py:508` 在 full/no/dropout 下复用同一组权重评估
  - `mpe_repro/plotting.py:29` 及同文件其余函数出图
  - `run_comm.py:487` 生成 batch 报告
- Flow B: 单步仿真内核
  - `mpe_repro/simulator.py:292` `_step`
  - `mpe_repro/graph_switch.py:55` 动态目标交换
  - `mpe_repro/comm_graph.py:18` 生成同组通信邻接
  - `mpe_repro/controller.py:144` 计算 pursuer/evader 控制
  - `mpe_repro/dynamics.py:57` RK4 推进一步
- Flow C: 机制解释 demo
  - `run_collision_demo.py:30` 构造“必须交叉”的场景
  - `run_cross_demo.py:30` / `run_cross_demo.py:139` 用更简化案例展示有无通信时轨迹差异

## 5. Core Modules And Roles
| Module | Role | Depends On | Used By | Why It Matters |
|---|---|---|---|---|
| `run_comm.py` | 项目主入口，组织 batch 训练/评估/出图/汇总 | `config`, `general_scenarios`, `simulator`, `plotting`, `report` | 用户直接运行 | 看这个文件就能理解整个实验流程 |
| `mpe_repro/simulator.py` | 真正的训练与评估编排器 | `controller`, `dynamics`, `features`, `graph_switch`, `offpolicy_ls`, `comm_graph` | `run_comm.py` 和 demo 脚本 | 这是运行链路的中心枢纽 |
| `mpe_repro/controller.py` | V-SNAC 控制律 + 通信形成梯度注入 | `dynamics`, `features` | `simulator.py` | 这里定义了“论文复现”和“作者扩展”如何合并 |
| `mpe_repro/graph_switch.py` | 动态目标交换图 | NumPy | `simulator.py` | 对应原论文的动态 target assignment |
| `mpe_repro/comm_graph.py` | 同组通信拓扑、dropout、编队位移差 | NumPy | `simulator.py` | 这是作者新增通信层的基础设施 |
| `mpe_repro/offpolicy_ls.py` | critic 最小二乘更新器 | NumPy | `simulator.py` | 体现了实现层面对论文 PI / V-SNAC 的近似求解 |
| `mpe_repro/general_scenarios.py` | 泛化场景生成器 | `config` | `run_comm.py` | 让工程不只停在论文固定场景上 |
| `mpe_repro/dynamics.py` | 6-DOF 非线性飞行器动力学 + RK4 | `config` | `simulator.py`, `controller.py` | 决定实验不是简单双积分器，而是更接近原论文的飞行器模型 |
| `mpe_repro/features.py` | 6 维 V-SNAC basis 与 Jacobian | `config` | `controller.py`, `simulator.py` | 与原论文 value approximation 直接对应 |
| `mpe_repro/report.py` | JSON/Markdown 报告和摘要指标 | `simulator.py` dataclasses | `run_comm.py` | 实验结论基本都从这里导出 |

## 6. Layering / Boundaries
- Interface: `run_comm.py`, `run_collision_demo.py`, `run_cross_demo.py`
- Application: `mpe_repro/simulator.py`, `mpe_repro/report.py`, `mpe_repro/plotting.py`
- Domain: `mpe_repro/controller.py`, `mpe_repro/graph_switch.py`, `mpe_repro/comm_graph.py`, `mpe_repro/offpolicy_ls.py`
- Infrastructure / Math primitives: `mpe_repro/dynamics.py`, `mpe_repro/features.py`, `mpe_repro/config.py`
- Boundary notes: 这个项目的层次其实比较清楚，问题不在“耦合过重”，而在“同一理论想法有两种并行表述”，导致文档和实现之间需要额外辨认。

## 7. 复现了什么，自己加了什么
### 复现原论文的部分
- 6-DOF 非线性动力学和饱和控制结构：`mpe_repro/dynamics.py:8`, `mpe_repro/controller.py:181`, 原论文 PDF 第 3-5 页。
- V-SNAC 的六维 basis：`mpe_repro/features.py:8`, `PAPER_COMM_VSNAC.tex:119`
- pursuer/evader 的 `tanh` 饱和控制律：`mpe_repro/controller.py:171`, 原论文 PDF 第 4-5 页公式 (19)/(22)。
- 动态目标交换：`mpe_repro/graph_switch.py:55`, 原论文 PDF 第 3-4 页 Algorithm 1 / Theorem 1。

### 明显属于你自己的扩展
- 同组追逐者通信图，而不是只做 pursuer-evader target graph：`mpe_repro/comm_graph.py:18`, `PAPER_COMM_VSNAC.tex:152`
- 速度-位置耦合的 formation gradient / coordination potential：`mpe_repro/controller.py:21`, `PAPER_COMM_VSNAC.tex:158`
- dropout 通信鲁棒性评估：`run_comm.py:264`, `mpe_repro/comm_graph.py:42`, `PAPER_COMM_VSNAC.tex:213`
- `d_safe` 自适应放大项：`mpe_repro/controller.py:65`, `mpe_repro/config.py:126`, `0331Report.md:20`
- 受控交叉/碰撞 demo，用来专门展示“安全性收益”：`run_collision_demo.py:1`, `run_cross_demo.py:1`

## 8. 实验结果现在说明了什么
- 当前输出更强地支持“通信层提升安全间距/编队分离”，不强支持“通信层总能显著加快捕获”。例如 `3v1` 里 full comm 的 `min_d_min` 从 135.6 m 提升到 282.2 m，但 capture time 反而从 47.15 s 变成 55.05 s（`outputs/comm_structured/REPORT.md:11`, `outputs/comm_structured/3v1/summary.json:90`）。
- `6v3` 更符合你的论文叙事：full comm 下 capture time 稍优于 no comm，`d_min` 也从 100.7 m 提升到 123.0 m，dropout 结果又介于两者之间（`outputs/comm_structured/REPORT.md:14`, `outputs/comm_structured/6v3/summary.json:129`）。
- `3v3` 三种模式完全相同，说明“每个逃逸者只有一名追逐者时通信协调层自然失活”这个退化性质在代码输出里确实成立（`outputs/comm_structured/REPORT.md:12`）。

## 9. Extension Points
- 新增场景规模/布局：改 `GeneralScenarioSpec` 和 `build_general_scenario()`，这是最安全的扩展点（`mpe_repro/general_scenarios.py:15`, `mpe_repro/general_scenarios.py:210`）。
- 新增通信拓扑规则：改 `CommunicationGraph.build_adjacency()`，比如从“同组全连接”变成距离阈值图（`mpe_repro/comm_graph.py:18`）。
- 新增协调律：改 `formation_gradient()` 即可，不需要动 critic 训练流程（`mpe_repro/controller.py:21`）。
- 新增学习法：可以替换 `ReplayLeastSquares.solve()`，但这会影响整个 critic 收敛行为（`mpe_repro/offpolicy_ls.py:53`）。
- 新增论文图表：通常只需要在 `plotting.py` 加绘图函数，再在 `run_comm.py:291` 后挂接。

## 10. Risks / Design Debt
- Risk: 文档和代码对“训练时是否开启通信”表述不一致。
  - Evidence: `mpe_repro/simulator.py:420` 明确写着训练强制 `gamma = 0`；但 `run_comm.py:246` 注释写的是 “Train with full communication”，生成的批报告也写了 “Training is performed ONCE with full communication” (`outputs/comm_structured/REPORT.md:18`)。
  - Impact: 很容易误读实验方法，尤其是在写论文或答辩时。
  - Mitigation: 统一措辞，明确为“用 baseline V-SNAC 训练，同一组权重在多种通信模式下评估”。
- Risk: 仓库里同时存在“增广误差状态”和“增广值函数”两套理论表述。
  - Evidence: `COMM_VSNAC_DERIVATION.tex:92` 走的是增广误差状态路线；`PAPER_COMM_VSNAC.tex:168` 和 `mpe_repro/controller.py:175` 走的是增广值函数/势函数梯度路线；`mpe_repro/comm_graph.py:86` 的 `augmented_error()` 在代码里没有被主流程调用。
  - Impact: 后续读者会分不清当前结果到底对应哪套理论。
  - Mitigation: 明确标注哪份推导是草稿、哪份才是当前实现对应的定稿。
- Risk: 研究结论应聚焦“安全性/分离性收益”，不要过度声称“追踪性能全面更优”。
  - Evidence: `outputs/comm_structured/REPORT.md:11` 到 `outputs/comm_structured/REPORT.md:15` 中，多个 case 的 mean assigned error / capture time 并不是始终更好。
  - Impact: 如果论文表述过满，容易被审稿人抓住反例。
  - Mitigation: 把主结论聚焦到 `d_min`、formation error、dropout graceful degradation。

## 11. Learning Roadmap
### 15-minute orientation
1. 先看 `0331Report.md:5`，理解项目为什么要在原论文上加通信层。
2. 再看 `run_comm.py:229` 和 `run_comm.py:576`，掌握怎么跑主实验。
3. 最后看 `mpe_repro/simulator.py:64`，建立训练/评估主链路的整体图像。

### 2-hour deep understanding
1. 读 `mpe_repro/controller.py:21` 和 `mpe_repro/controller.py:144`，搞清楚控制律。
2. 读 `mpe_repro/graph_switch.py:55`，理解动态 target swap。
3. 读 `mpe_repro/offpolicy_ls.py:16`，理解 critic 如何更新。
4. 对照 `PAPER_COMM_VSNAC.tex:145`，把理论写法和代码实现一一对应起来。

### 1-day first contribution path
1. 先修正文档措辞，把“训练使用 full communication”的表述改准确。
2. 给“当前实现采用哪套理论表述”补一份简洁说明，避免未来自己也混淆。
3. 再尝试做你周报里提到的自适应 `gamma` 扩展，因为代码里已经有 `d_safe` 放大入口，落地成本相对最低（`mpe_repro/controller.py:65`, `mpe_repro/config.py:126`）。

## 12. Assumptions / Open Questions
- Assumption: 当前最权威的“实现对应理论文档”应当是 `PAPER_COMM_VSNAC.tex`，而不是 `COMM_VSNAC_DERIVATION.tex`。理由是它和 `controller.py` 的实现方式一致。
- Question: 论文 Xu 2024 原文确实用了动态图增强 team cohesion，但当前项目是否还需要进一步做“与原文计算负担 31% 降低”的严格对比实验？从 `0331Report.md:5` 看，你已经意识到这条线不好讲。
- Question: `ReplayLeastSquares` 的参数名是 `n_evaders`，但训练里实际按 `n_pursuers` 个 critic 使用（`mpe_repro/simulator.py:424`, `mpe_repro/offpolicy_ls.py:33`）；这只是命名沿袭，还是早期实现残留，需要后续统一。

## 13. Evidence Index
- `run_comm.py:229`: 单 case 训练/评估/汇总主流程入口。
- `run_comm.py:264`: 同一组权重在 `full_comm` / `no_comm` / `dropout` 三种模式下评估。
- `run_comm.py:576`: 命令行主入口。
- `mpe_repro/simulator.py:420`: 训练阶段固定 `gamma = 0`。
- `mpe_repro/simulator.py:508`: 评估阶段允许覆盖通信参数。
- `mpe_repro/controller.py:21`: formation gradient 的解析定义。
- `mpe_repro/controller.py:144`: 当前实际 pursuer/evader 控制律。
- `mpe_repro/comm_graph.py:18`: 同组通信图构造。
- `mpe_repro/comm_graph.py:86`: 未进入主流程的 `augmented_error()`。
- `mpe_repro/graph_switch.py:55`: 动态目标交换算法。
- `mpe_repro/offpolicy_ls.py:16`: critic 的 off-policy LS 更新。
- `mpe_repro/general_scenarios.py:210`: 通用场景生成器。
- `0331Report.md:7`: 作者新增通信协调层的直接动机。
- `PAPER_COMM_VSNAC.tex:52`: 论文化后的三层框架总述。
- `PAPER_COMM_VSNAC.tex:145`: 通信增广控制理论部分。
- `outputs/comm_structured/REPORT.md:11`: 现有实验汇总结果。
