# 本周进展（2026-03-24）

## 总览
本周工作围绕两条主线展开：

1. 在三追一场景下，引入考虑追逐者组内通信的组级价值函数，并验证通信中断/恢复情况下系统仍可重新收敛。
2. 在原有追逃框架基础上，完成从固定 benchmark 场景（3v1、3v3）向一般 `m` 对 `n` 场景的推广，实现任意 `m > n` 的多追多逃仿真、批量运行与并行评测，并补充 `5v3` 作为代表性扩展场景。

---

## 一、通信感知价值函数修改（多追一）

### 1. 工作目标
原始三追一设置中，每个追逐者主要从各自视角构造价值函数并求控制。为了体现同一目标下追逐者之间的组内协同，本周进一步引入了组级建模思路：

- 将多个追逐者视为一个合作追捕小组；
- 组内共享同一个 team critic；
- 控制律仍保留饱和输入形式，但由组级状态误差驱动；
- 额外构造通信中断/恢复场景，验证通信缺失对团队追捕性能的影响。

### 2. 已完成内容
对应工程目录：
- [repro_vsnac_mpe_commgroup](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup)

已完成的主要工作包括：
- 建立三追一场景下的组级状态表示与 team critic；
- 在执行阶段为每个追逐者维护本地组状态估计，用于模拟通信不完整情况下的信息失配；
- 支持通信中断/恢复调度，并用估计误差图展示断联期间的局部团队状态失配；
- 补充组级价值函数、Hamiltonian、稳定性与一致最终有界（UUB）分析文档；
- 输出 no-drop 与 drop/recovery 两组对照实验结果。

### 3. 代表性结果
稳定结果目录：
- [team_comm_final_converged1](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/outputs/team_comm_final_converged1)

对应汇总文件：
- [summary.json](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/outputs/team_comm_final_converged1/summary.json)
- [REPORT.md](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/outputs/team_comm_final_converged1/REPORT.md)

关键结果如下：

- 无通信中断（no-drop）捕获时间：`26.6 s`
- 通信中断后恢复（drop/recovery）捕获时间：`45.4 s`
- 在相同对比时刻 `26.6 s`：
  - `no-drop Eteam = 408.689`
  - `drop/recovery Eteam = 575.565`
- 通信恢复后最终估计误差重新回到 `0`

### 4. 本周结论
这一部分说明：
- 组级价值函数建模在三追一场景中是可行的；
- 通信中断会显著拖慢追捕过程，并提高团队误差；
- 恢复通信后，局部组状态估计可重新同步，系统重新回到收敛邻域；
- 从实验和推导两方面，均能支撑“通信信息对协同追捕有效性有直接影响”。

### 5. 文档沉淀
相关推导文档已补齐：
- [TEAM_COMM_DERIVATION_MATHJAX_CN.md](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/TEAM_COMM_DERIVATION_MATHJAX_CN.md)
- [TEAM_COMM_DERIVATION_OVERLEAF_CN.tex](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/TEAM_COMM_DERIVATION_OVERLEAF_CN.tex)
- [TEAM_COMM_STABILITY_DETAIL_MATHJAX_CN.md](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/TEAM_COMM_STABILITY_DETAIL_MATHJAX_CN.md)
- [TEAM_COMM_STABILITY_DETAIL_OVERLEAF_CN.tex](/e:/毕业设计/mpe_game/repro_vsnac_mpe_commgroup/TEAM_COMM_STABILITY_DETAIL_OVERLEAF_CN.tex)

---

## 二、从 benchmark 场景推广到一般 m 对 n 场景

### 1. 工作目标
为了避免实验只停留在 `3v1` 和 `3v3` 的固定 benchmark，本周将原框架推广到一般 `m` 个追逐者对 `n` 个逃避者的设置，重点关注 `m > n` 情况下的多追多逃仿真与验证能力。

目标不只是“能跑更多智能体”，还包括：
- 场景自动生成；
- 统一输入接口；
- 动态图与固定图的公平对照；
- 支持批量运行与并行执行；
- 输出更适合大规模实验汇报的统一指标与可视化结果。

### 2. 已完成的计算机实现增强
对应工程目录：
- [repro_vsnac_mpe_mn](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn)

本周新增的更偏“计算机实现”的能力包括：

#### 2.1 一般化场景生成
- 支持任意 `m` 个追逐者、`n` 个逃避者；
- 支持通过命令行直接指定单个 case，例如 `5v3`；
- 支持 case 列表，例如 `3x1 3x3 5x3 6x4`；
- 支持 sweep 多组 `m > n` 组合。

#### 2.2 批量与并行运行
- 支持 `--parallel-workers k` 多进程并行；
- 支持 `--parallel-workers 0` 自动根据 CPU 核数和任务数选择 worker 数；
- 适合做一批 `m > n` 组合的快速比较。

#### 2.3 更统一的结果表达
- 对每个 pursuer 统一输出分配目标下的总体残差范数，而不是只看 `x/y/h` 单通道；
- 支持 3D 轨迹 GIF 动图导出；
- 保留动态分配与固定分配的同权重、同初值、同 seed 对照。

### 3. 关键代码位置
- 通用运行入口：[run_generalized.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/run_generalized.py)
- 一般化场景生成器：[general_scenarios.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/mpe_repro/general_scenarios.py)
- 结果绘图工具：[plotting.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/mpe_repro/plotting.py)
- 使用说明：[README.md](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/README.md)

### 4. 已验证的代表性场景

#### 4.1 标准场景：3v1、3v3
已保留并继续作为 benchmark：
- [repro_20260311_003452](/e:/毕业设计/mpe_game/repro_vsnac_mpe/outputs/repro_20260311_003452)
- [repro_20260311_011233](/e:/毕业设计/mpe_game/repro_vsnac_mpe/outputs/repro_20260311_011233)

这一部分用于说明：
- `3v1` 下基础非线性追捕控制有效；
- `3v3` 下动态图目标分配比固定图更利于降低平均分配误差和缩短捕获时间；
- 两次独立输出可相互验证结果稳定性。

#### 4.2 扩展场景：5v3
正式结果目录：
- [generalized_5v3_resnorm_gif/5v3](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/generalized_5v3_resnorm_gif/5v3)

对应汇总文件：
- [summary.json](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/generalized_5v3_resnorm_gif/5v3/summary.json)
- [REPORT.md](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/generalized_5v3_resnorm_gif/5v3/REPORT.md)

关键结果：
- 动态图捕获时间：`30.45 s`
- 固定图捕获时间：`33.10 s`
- 动态图 mean assigned error：`358.08`
- 固定图 mean assigned error：`401.24`
- V-SNAC critic 数量：`5`
- 估计 AC 网络数：`16`
- 估计网络减少比例：`68.75%`

说明：在 `5v3` 设置下，动态图仍然能够在相同训练权重、相同初值、相同评估随机种子下，取得更快的捕获速度和更低的平均分配误差。

#### 4.3 一般 m>n 快速扫例
快速 sweep 目录：
- [mn_sweep_quick_463](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_sweep_quick_463)

已完成的 case 包括：
- `4v2`
- `4v3`
- `5v2`
- `5v3`
- `6v2`
- `6v3`

对应 sweep 汇总文件：
- [SWEEP_SUMMARY_2026_03_24.md](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_sweep_quick_463/SWEEP_SUMMARY_2026_03_24.md)

已核对的 quick 结果如下：

| case | dynamic capture (s) | fixed capture (s) | dynamic mean err | fixed mean err |
| --- | ---: | ---: | ---: | ---: |
| 4v2 | 30.90 | 32.55 | 456.05 | 610.84 |
| 4v3 | 31.10 | 38.15 | 439.44 | 509.80 |
| 5v2 | 32.50 | 37.95 | 464.48 | 590.52 |
| 5v3 | 33.85 | 36.45 | 451.44 | 505.14 |
| 6v2 | 33.90 | 39.60 | 476.87 | 671.95 |
| 6v3 | 31.15 | 36.35 | 441.20 | 566.82 |

这说明：
- 从 `4v2` 扩展到 `6v3` 的多组 `m > n` quick case 中，动态图分配均比固定图具有更快的捕获时间；
- 同时平均分配误差也保持更优；
- 因而“动态图机制不仅对 benchmark 有效，对更一般的 `m > n` 场景也具有延展性”。

### 5. 本周结论
这一部分工作的核心进展是：
- 已将原有固定 benchmark 工作流推广为一般 `m` 对 `n` 仿真框架；
- 已支持批量 case 生成、并行执行、统一汇总和大规模可视化；
- 已通过 `5v3` 和一组 `m > n` quick cases 验证动态图分配的可扩展有效性；
- 从方法层面看，当前系统已不再局限于 `3v1` / `3v3` 的特定展示，而具备继续扩展到更大规模实验的基础。

---

## 三、本周工作总结
本周的两个任务分别从“控制建模”和“实验平台”两个角度推进了课题：

- **在控制建模层面**：完成了面向多追一场景的通信感知组级价值函数建模，并验证了通信中断/恢复下的性能变化；
- **在实验平台层面**：完成了从 benchmark 到一般 `m > n` 场景的通用化、批量化与并行化实现，并用 `5v3` 和多组 quick case 给出扩展结果。

这两条线共同说明：
- 课题不仅能在标准场景下得到结果；
- 也具备进一步扩展到更一般规模、更偏工程验证的多智能体追逃实验能力。

---

## 四、下周建议
下周可继续推进的方向包括：

1. 继续补充更大规模 `m > n` 场景，例如 `6v4`、`8v5`；
2. 将 quick sweep 的批量结果进一步整理成统一表格或曲线；
3. 在通信感知价值函数上继续加入组内编队/接力机制；
4. 将当前 generalized 结果进一步整理进会议稿正文和图表页中。

