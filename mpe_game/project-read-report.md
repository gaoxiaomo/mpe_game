# MPE 三个项目阅读报告

## 1. 结论先看

这三个项目已经形成了一条比较清晰的演进链：

1. `repro_vsnac_mpe` 是论文《Approximate Optimal Strategy for Multiagent System Pursuit-Evasion Game》的基础复现版，核心目标是复现论文中的非线性飞行器动力学、动态图切换、V-SNAC 和 `3v1 / 3v3` 场景。
2. `repro_vsnac_mpe_mn` 是在复现版之上做的通用化与计算优化版，重点从“复现论文图”转向“支持一般 `m` 对 `n` 场景、批量评测与加速”。
3. `repro_vsnac_mpe_commgroup` 是另一条扩展线，不再重点研究多追多换目标，而是研究多追一中的组级 critic 与通信信息结构。

整体上，你的工作不是简单地做了三个互不相关的工程，而是已经有了：

- 一条“论文复现 -> 通用化 -> 算法工程优化”的主线
- 一条“论文复现 -> 多追一通信协同”的副线

这对后续写毕业论文是有利的，因为结构很自然。

## 2. 论文核心方法与你代码的对应

论文的核心点有四个：

- 非线性飞行器动力学与受限输入控制  
  证据：`paper_text_fitz/page_09.txt:19-66`
- 动态目标图 Algorithm 1，用 pairwise swap 改善团队凝聚  
  证据：`paper_text_fitz/page_03.txt:149-232`, `paper_text_fitz/page_04.txt:1-45`
- 用 HJI 推导出的饱和 `tanh` 型最优控制律  
  证据：`paper_text_fitz/page_05.txt:1-64`
- 用 V-SNAC / off-policy RL 近似值函数，降低多智能体下的计算负担  
  证据：`paper_text_fitz/page_07.txt:45-114`, `paper_text_fitz/page_08.txt:1-66`, `paper_text_fitz/page_11.txt:28-36`

你在基础复现版中的对应关系是比较完整的：

- 动力学：`repro_vsnac_mpe/mpe_repro/dynamics.py:8`
- 控制律与阶段代价：`repro_vsnac_mpe/mpe_repro/controller.py:21`
- 动态目标图：`repro_vsnac_mpe/mpe_repro/graph_switch.py:6`
- LS 版 off-policy critic 更新：`repro_vsnac_mpe/mpe_repro/offpolicy_ls.py:16`
- 训练与评估主循环：`repro_vsnac_mpe/mpe_repro/simulator.py:59`
- 实验入口：`repro_vsnac_mpe/run_repro.py:106`

## 3. 三个项目分别在做什么

### 3.1 `repro_vsnac_mpe`

这是“论文复现主工程”。

它的工作方式是：

- 用 `ScenarioConfig` 固定论文中的 `3v1` 和 `3v3` 初始条件  
  证据：`repro_vsnac_mpe/mpe_repro/config.py:125-207`
- 用 `MPESimulator.train_policy()` 做 critic 训练  
  证据：`repro_vsnac_mpe/mpe_repro/simulator.py:353-432`
- 用 `evaluate_policy()` 做固定图/动态图评估  
  证据：`repro_vsnac_mpe/mpe_repro/simulator.py:434-549`
- 在 `run_repro.py` 中生成论文式图像和报告  
  证据：`repro_vsnac_mpe/run_repro.py:278-497`

我对它的判断：

- 它已经不只是“能跑”，而是带有比较完整的论文图复现实验脚本。
- 它的优势是结构清楚、场景固定、便于论文对照。
- 它的局限是大量参数和场景是写死的，后续想扩展到一般规模时不够灵活。

### 3.2 `repro_vsnac_mpe_mn`

这是“通用化 + 计算优化主工程”。

相比基础版，它新增了三类东西：

- 一般 `m,n` 场景生成器  
  证据：`repro_vsnac_mpe_mn/mpe_repro/general_scenarios.py:15-298`
- 向量化动态图更新与向量化 pairwise error  
  证据：`repro_vsnac_mpe_mn/mpe_repro/graph_switch.py:45-115`, `repro_vsnac_mpe_mn/mpe_repro/simulator.py:231-233`
- 批量 case、并行执行、runtime 统计  
  证据：`repro_vsnac_mpe_mn/run_generalized.py:307-472`, `repro_vsnac_mpe_mn/run_generalized.py:537-695`

这条线的结果也已经比较稳定：

- 稳定结果总结里，`3v3/4v2/4v3/5v2/5v3/6v2/6v3` 都给出了动态图与固定图对照  
  证据：`STABLE_RESULTS_2026_03_24_CN.md:16-31`
- 你自己也已经把“通用化、网络压缩、向量化、并行”整理成论文可写的优化点  
  证据：`MN_OPTIMIZATION_FOR_PAPER_2026_03_24_CN.md:10-18`, `MN_OPTIMIZATION_FOR_PAPER_2026_03_24_CN.md:108-176`, `MN_OPTIMIZATION_FOR_PAPER_2026_03_24_CN.md:180-233`

我对它的判断：

- 这是目前最适合继续往“论文创新点/工程贡献”上推进的一版。
- 它已经不只是复现，而是在做“计算机实现层面的扩展与优化”。
- 如果你的毕业论文要突出“系统实现能力”和“规模扩展能力”，这条线最强。

### 3.3 `repro_vsnac_mpe_commgroup`

这是“多追一组级通信版本”。

它和 `mn` 不是同一扩展方向，而是另一种问题重写：

- 不再主要研究多追多时的目标切换
- 而是把多追一写成组状态 `Z` 上的 team critic
- 再在执行阶段人为构造 `full / local-only / dropout` 三种信息结构

新增核心模块是：

- 组级场景与 blackout 窗口：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_config.py:32-159`
- 组级特征：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_features.py:10-116`
- 组级控制器：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_controller.py:28-159`
- 组级 LS 求解：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_ls.py:16-78`
- 组级训练/评估：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_simulator.py:68-478`

它的论文化表达也已经成形：

- 你把通信的价值明确表述成“信息结构扩大后，可选策略集合更大”  
  证据：`COMMUNICATION_SIGNIFICANCE_2026_03_24_CN.md:53-139`
- 你也明确写了实验上不要求每个小场景都严格更优，而是看规模上升后的趋势  
  证据：`COMMUNICATION_SIGNIFICANCE_2026_03_24_CN.md:143-162`
- 稳定结果显示 `4v1` 和 `5v1` 的通信优势明显，`3v1` 则不明显  
  证据：`COMMUNICATION_SIGNIFICANCE_2026_03_24_CN.md:238-250`

我对它的判断：

- 这条线理论叙事是成立的，而且和原论文“动态图换目标”方向不同，容易形成你自己的创新小节。
- 但它更像一个“针对多追一的组级重建模项目”，不建议和 `mn` 硬揉成一个算法主线。
- 更适合作为“第二创新点”或者“扩展研究”。

## 4. 三个项目之间的真实关系

从代码结构看，它们不是完全独立的：

- `repro_vsnac_mpe` 与 `repro_vsnac_mpe_commgroup` 的 `config.py / controller.py / dynamics.py / features.py / offpolicy_ls.py / report.py / simulator.py` 是完全相同的一套基础代码
- `repro_vsnac_mpe_mn` 则只在 `graph_switch.py / simulator.py / plotting.py` 上做了增量改造，并新增 `general_scenarios.py`

这说明你现在的仓库逻辑其实是：

- 先复制一份基础复现工程
- 然后在副本上继续改

短期这样很方便，但长期维护成本会越来越高。

## 5. 我认为最值得优先处理的优化点

下面按“优先级”给你。

### 优先级 A：通信版本的“估计”实现和文档表述不完全一致

当前 `team_comm_simulator.py` 在 blackout 时采用的是“零填充 + mask”：

- 不可见块直接不写入，保持为 0  
  证据：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_simulator.py:159-172`

但 `run_team_comm.py` 的报告里写的是 “stale teammate-relative blocks”：

- 证据：`repro_vsnac_mpe_commgroup/run_team_comm.py:76-81`

也就是说：

- 你的理论与稳定结果文档主要在讲“零填充 + mask”
- 但单案例报告文字又说成了“使用陈旧估计”
- 而且 `_single_step()` 虽然接收 `team_estimates`，实际上没有把上一时刻估计继续传播进去  
  证据：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_simulator.py:174-217`

建议：

1. 如果你要的是“零填充 + mask”实验，就把所有文档统一成这个说法。
2. 如果你真正想研究“断联时持有最近一次队友估计”，那就把 `team_estimates` 真正用于 blackout 期间的持久化/预测更新。

这是我认为当前最重要的一个“概念一致性”问题。

### 优先级 A：三个项目的公共内核已经明显重复，建议抽出共享 core

你现在有大量完全重复的文件副本，这会带来三个问题：

- 一个 bug 修三遍
- 一个参数改三遍
- 文档和实现更容易跑偏

最直接的结构优化是：

- 新建一个共享包，例如 `mpe_core/`
- 把 `config / controller / dynamics / features / offpolicy_ls / report / baseline simulator` 抽出来
- `repro`、`mn`、`commgroup` 只保留各自新增的 runner 和 extension 模块

这个优化不一定马上提升算法性能，但会显著降低后面写论文、调参数、修图表的维护成本。

### 优先级 B：基础版和 mn 版里有一批“看起来可调、实际上没用”的参数

以下参数目前进入了配置与构造函数，但没有参与控制律或切换逻辑：

- `policy_gain`
- `k_pos_p`
- `k_vel_p`
- `k_pos_e`
- `k_vel_e`
- `max_switch_worsening`

证据：

- 参数定义：`repro_vsnac_mpe/mpe_repro/config.py:39-74`
- 被传入控制器：`repro_vsnac_mpe/mpe_repro/simulator.py:73-89`
- 但在控制器中只被保存，没有继续参与实际计算：`repro_vsnac_mpe/mpe_repro/controller.py:24-60`
- `max_switch_worsening` 进入 `graph_switch.update()`，但实现里被注释为“仅保留兼容性，不参与 gating”  
  证据：`repro_vsnac_mpe/mpe_repro/graph_switch.py:80-85`

建议：

1. 要么删除这些死参数，避免后续调参误判。
2. 要么明确把它们接回控制律/切换判据。

### 优先级 B：team communication 版的训练稳定化技巧值得反向迁移到基础版和 mn 版

`commgroup` 版的训练已经比基础版更“工程化”：

- 每轮重新构造 recent-window LS buffer  
  证据：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_simulator.py:274-305`
- 对最新样本加更大权重  
  证据：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_ls.py:58-67`
- 做有界投影  
  证据：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_ls.py:70-74`
- 做 backtracking 式接受判据  
  证据：`repro_vsnac_mpe_commgroup/mpe_repro/team_comm_simulator.py:312-343`

相比之下，基础版 / mn 版还是：

- 单个 replay 缓冲区跨所有 iteration 持续累计
- 固定 `alpha`
- 没有 validation metric 和回退机制  
  证据：`repro_vsnac_mpe/mpe_repro/simulator.py:357-423`

这说明你已经在 `commgroup` 线上摸索出一套更稳的训练工程技巧，建议后续把它反向迁移到 `repro` 和 `mn`。

### 优先级 C：通信版本的论文叙事要避免对 `3v1` 过度宣传

从稳定结果看：

- `3v1` 下 `full communication` 和 `local only` 差别很小，甚至 matched-time `Eteam` 上 `local only` 略低  
  证据：`COMMUNICATION_SIGNIFICANCE_2026_03_24_CN.md:238-259`
- 真正明显的优势是在 `4v1` 和 `5v1`  
  证据：`COMMUNICATION_SIGNIFICANCE_2026_03_24_CN.md:240-250`

所以写作上建议：

- 把通信版本的贡献表述成“规模增长后更明显的协同收益”
- 不要写成“通信在所有多追一场景都严格更优”

你当前的说明文档其实已经意识到这点了，这个方向是对的。

### 优先级 C：基础复现版最好再补一个“动态图确实发生切换”的默认稳定出口

基础版 `run_repro.py --quick` 我本地跑出来时：

- `3v3` quick 结果里 `switch_count = 0`  
  证据：`repro_vsnac_mpe/outputs/repro_20260324_140544/metrics_summary.json`

而稳定总结果里，`mn` 版本的 `3v3`、`4v2`、`5v3` 等场景都已经能稳定出现一次切换：

- 证据：`STABLE_RESULTS_2026_03_24_CN.md:19-26`

这意味着：

- 你的动态图算法本身是能工作的
- 但基础版默认 quick 演示不一定能把这一点展示出来

建议后续给基础版补一个“演示型默认参数组合”，让用户一跑就能看到 switch。

## 6. 如果按论文推进，建议怎么分主次

我建议你把三条内容这样组织：

### 主线

`repro_vsnac_mpe` -> `repro_vsnac_mpe_mn`

可以写成：

- 先复现论文方法
- 再扩展到一般 `m` 对 `n`
- 再做向量化和并行化计算优化

这条线最完整，也最像一篇“从理论复现到工程扩展”的毕业设计主线。

### 副线

`repro_vsnac_mpe_commgroup`

可以写成：

- 在多追一场景下，将原本局部 critic 形式改造成组级 critic
- 用信息结构差异来研究通信的作用
- 重点强调规模增加后通信的协同收益

这条线更适合放成“扩展研究”或“第二创新点”。

## 7. 我建议你下一步优先做什么

如果你要继续优化，我建议顺序是：

1. 先统一 `commgroup` 的“零填充 / stale estimate”文档与实现。
2. 再抽共享 core，减少三套副本。
3. 把 `commgroup` 的稳定训练技巧迁回 `repro` 和 `mn`。
4. 最后再决定是否继续深化通信版本，还是把重心完全放到 `m` 对 `n` 通用化这条线。

## 8. 附：本次快速运行观察

我额外做了 quick 烟测：

- `repro_vsnac_mpe/run_repro.py --quick` 成功
- `repro_vsnac_mpe_mn/run_generalized.py --quick --parallel-workers 2 --skip-plots` 成功
- `repro_vsnac_mpe_commgroup/run_team_comm.py --quick` 成功
- `repro_vsnac_mpe_commgroup/run_comm_vs_local.py --quick --parallel-workers 2` 虽然在交互超时前返回较慢，但结果文件已完整写出

说明三个工程目前都处于“可以继续做实验”的状态，不是那种需要先大修才能继续推进的仓库。
