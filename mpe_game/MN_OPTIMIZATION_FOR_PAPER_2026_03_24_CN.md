# m 对 n 通用追逃框架的计算优化

## 1. 目标与定位

当前这条工作线不再把重点放在通信机制上，而是聚焦于**多追多（m 对 n）非线性追逃系统的通用化实现与计算优化**。相比仅验证 `3v1` 和 `3v3` 两个标准场景，我们现在希望把系统扩展到任意 `m>n` 或一般 `m,n` 输入，并进一步回答两个更偏计算机系统的问题：

1. 在场景规模增大后，当前求解框架是否仍然可运行、可复用、可批量验证？
2. 是否可以通过更高效的实现方式，在**不改变算法结果**的前提下显著降低计算开销？

围绕这两个问题，目前已经形成了四个可以写进论文的优化点：

- 通用 `m` 对 `n` 场景生成与统一评测框架
- V-SNAC 结构下的网络数量压缩
- 动态目标图更新的向量化计算优化
- 面向批量验证的并行执行模式

下面分别说明。

---

## 2. 优化点一：从固定 benchmark 扩展到一般 m 对 n 场景

### 2.1 做了什么

在通用工程中，我们不再把仿真写死为 `3v1` 或 `3v3`，而是构造了一个通用场景生成器：

- 输入：追逐者数量 `m`、逃避者数量 `n`
- 输出：
  - 初始状态矩阵
  - 期望位移矩阵 `r_{j,i}`
  - 初始分配图
  - 捕获半径、仿真时长、交换阈值等参数

对应代码入口：

- [run_generalized.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/run_generalized.py)
- [general_scenarios.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/mpe_repro/general_scenarios.py)

这样做的意义是：

- 可以统一支持 `3v1`、`3v3`、`5v3` 等场景
- 可以进一步生成 `4v2`、`4v3`、`5v2`、`6v3` 等一般 `m` 对 `n` 场景
- 后续实验可以从“单个案例展示”扩展到“批量 sweep 验证”

### 2.2 为什么这是一个可写的创新点

这部分的创新点不在于改变追逃控制律本身，而在于将原本局部、固定的 benchmark 实现，扩展为一个**可配置、可批量、可复用的通用仿真平台**。这更偏向计算机实现与系统验证能力，适合写成：

> 本文实现了一个面向一般 `m` 对 `n` 多智能体追逃问题的统一仿真与评测框架，使得原先仅针对固定小规模场景的算法验证能够推广到任意规模组合，并支持批量自动实验。

---

## 3. 优化点二：网络数量压缩（V-SNAC 结构）

### 3.1 比较对象

在通用实现中，我们保留了 V-SNAC 这条思路：

- 每个 pursuer 使用一个 critic
- 因而 V-SNAC critic 数量为

$$
N_{\text{V-SNAC}} = m
$$

如果采用传统 AC 结构，可以按每个 pursuer 和每个 evader 分别配置 actor/critic 对进行粗略估计，则网络总数可写为

$$
N_{\text{AC}} \approx 2(m+n)
$$

对应代码：

- [report.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/mpe_repro/report.py)

代码中网络数量统计就是：

- `v_snac_networks = m`
- `ac_networks_estimated = 2(m+n)`

网络压缩比例定义为

$$
\eta_{\text{reduce}}
= \frac{N_{\text{AC}} - N_{\text{V-SNAC}}}{N_{\text{AC}}} \times 100\%
$$

### 3.2 结果示例

在当前稳定批量结果中：

- `3v3`：`3` vs `12`，压缩 `75.00%`
- `5v3`：`5` vs `16`，压缩 `68.75%`
- `6v3`：`6` vs `18`，压缩 `66.67%`

稳定结果目录：

- [mn_batch_stable_20260324](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_batch_stable_20260324)

### 3.3 可写成论文的点

这部分可以直接写成：

> 在一般 `m` 对 `n` 场景下，本文实现沿用了 V-SNAC 结构，使 critic 数量仅随 pursuer 数量线性增长，而不需要像传统 AC 结构那样随 pursuer 和 evader 的总数成倍增加。这为大规模批量验证提供了更低的模型维护成本。

---

## 4. 优化点三：动态图更新的向量化计算

### 4.1 原始问题

在 `m` 对 `n` 场景中，动态图更新需要频繁计算 pursuer–evader 的加权距离矩阵。若使用逐项循环，则每次更新都需要进行大量重复的范数计算。

原始标量形式可理解为：

$$
g_{j,i} = \left\| \nu \odot \left(x_j^p - x_i^e + r_{j,i}\right) \right\|_2
$$

对于所有 `j=1,\dots,m` 和 `i=1,\dots,n`，如果逐项循环计算，代价较高；进一步，在 pairwise swap 更新时，这些距离还会被反复调用。

### 4.2 现在的实现

我们把这一步改成了**一次性广播构造 pairwise cost matrix**：

$$
G = \Big[g_{j,i}\Big]_{m\times n}
$$

对应代码：

- [graph_switch.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/mpe_repro/graph_switch.py)
- [simulator.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/mpe_repro/simulator.py)

其中：

- `DynamicTargetGraph.weighted_distance_matrix(...)` 统一生成 `m×n` cost matrix
- `DynamicTargetGraph.update(...)` 直接复用矩阵条目进行 swap 判据比较
- `team_error(...)` 支持传入已计算的 `pairwise_costs`，避免重复计算
- `MPESimulator._pairwise_errors(...)` 同样改成广播实现

### 4.3 正确性验证

我们专门做了“旧版标量核 vs 新版向量化核”的严格对照，要求：

1. 距离矩阵元素逐项一致
2. 动态图更新后的 assignment 完全一致

benchmark 输出目录：

- [benchmark_mn_compute_20260324](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/benchmark_mn_compute_20260324)

在 `summary.json` 中：

- `matrix_max_abs_diff = 0.0`
- `assignments_equal = true`

说明向量化实现没有改变原算法行为。

### 4.4 速度收益

在 `300` 次 microbenchmark 下，得到：

| Case | 距离矩阵 speedup | 动态图更新 speedup |
|---|---:|---:|
| 5v3 | 7.71x | 7.98x |
| 8v4 | 9.24x | 11.83x |
| 12v6 | 23.47x | 12.56x |

可见，随着规模增大，向量化内核的优势更明显。

### 4.5 可写成论文的点

这部分可以写成：

> 为提升一般 `m` 对 `n` 场景下动态图更新的效率，本文将 pursuer–evader 配对代价的逐项循环计算改写为向量化 pairwise cost matrix 计算，并在目标交换判据中复用该矩阵，从而在不改变分配结果的前提下显著降低了动态图更新开销。

---

## 5. 优化点四：面向批量验证的并行执行模式

### 5.1 为什么需要并行

当场景从 `3v1 / 3v3` 扩展到一般 `m` 对 `n` 后，单次实验已经不够。为了更有说服力，需要在多组场景上做 sweep，例如：

- `4v2`
- `4v3`
- `5v2`
- `5v3`
- `6v2`
- `6v3`

这时候总运行时间会快速上升，因此单进程串行评估会成为瓶颈。

### 5.2 我们怎么做的

在 [run_generalized.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/run_generalized.py) 中，批量 case 支持通过 `ProcessPoolExecutor` 并行运行。

同时，为了让 benchmark 更真实地反映仿真阶段的吞吐能力，我们加入了：

- `--skip-plots`：关闭绘图输出，仅保留仿真和统计

这样可以把绘图和文件 I/O 的干扰降到最低，更公平地测量批量仿真本身的计算效率。

另外，当前入口已经支持：

- `--parallel-workers k`：手动设置 worker 数
- `--parallel-workers 0`：自动使用一个合理的 worker 数（按 CPU 核数推断）

### 5.3 并行 benchmark 结果

仍然使用上面的 benchmark 目录：

- [benchmark_mn_compute_20260324](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/benchmark_mn_compute_20260324)

在关闭绘图的条件下，对 `4v2, 4v3, 5v2, 5v3, 6v2, 6v3` 这 6 个 case 做比较：

- serial wall time: `57.204 s`
- parallel wall time: `23.468 s`
- observed speedup: `2.438x`

同时：

- mean serial ms/step: `1.6362`
- mean parallel ms/step: `1.2155`

这说明在多 case 批量验证中，并行执行确实可以显著降低总等待时间。

### 5.4 可写成论文的点

这部分可以写成：

> 针对一般 `m` 对 `n` 场景需要进行大量参数组合验证的问题，本文实现了基于多进程的批量实验执行模式，并通过关闭绘图输出将 benchmark 聚焦于纯仿真阶段。在 6 个典型 case 的批量验证中，并行执行相较串行执行获得了约 `2.44x` 的 wall-time 加速。

---

## 6. 有效性结果：m 对 n 场景下动态图仍然有效

除了“算得更快”，还需要证明通用化后的框架仍然保持有效性。当前稳定结果目录：

- [mn_batch_stable_20260324](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_batch_stable_20260324)

代表性结果如下。

### 6.1 3v3

- dynamic capture: `34.25 s`
- fixed capture: `36.35 s`
- dynamic mean assigned error: `673.653`
- fixed mean assigned error: `886.503`

### 6.2 5v3

- dynamic capture: `36.15 s`
- fixed capture: `41.25 s`
- dynamic mean assigned error: `682.256`
- fixed mean assigned error: `769.438`

### 6.3 6v3

- dynamic capture: `38.85 s`
- fixed capture: `45.45 s`
- dynamic mean assigned error: `711.312`
- fixed mean assigned error: `915.140`

由这些结果可以说明：

- 通用化并没有破坏原有动态图机制
- 在 `n>1` 的多目标场景下，动态图仍然能缩短捕获时间
- 并且通常还能降低平均分配误差

因此，当前这条线的论文叙事可以比较自然地写成：

> 本文不仅在标准 `3v1`、`3v3` 场景中验证了方法有效性，还将其推广到了更一般的 `m` 对 `n` 多智能体追逃问题，并通过多组规模扩展实验验证了动态图机制与计算优化在更大场景中的可用性与效率优势。

---

## 7. 当前最适合写进论文的“优化点”总结

如果只保留最有力的、最偏计算机方向的 3 个点，我建议写成下面这三个：

### 优化点 A：通用 m 对 n 场景生成与批量验证框架
关键词：`generalization`、`scalability`、`benchmark automation`

### 优化点 B：V-SNAC 结构下的网络数量压缩
关键词：`network reduction`、`critic count scaling`

### 优化点 C：向量化动态图更新 + 并行批量执行
关键词：`vectorized kernel`、`parallel evaluation`、`runtime speedup`

这样结构上会比较紧：

- A 解决“能不能扩展到一般规模”
- B 解决“网络规模会不会爆炸”
- C 解决“算得够不够快”

---

## 8. 与其他方式的对比应该怎么写

### 8.1 和传统 AC 结构的对比
对比指标：网络数量

- V-SNAC: `m`
- AC estimated: `2(m+n)`

这是结构复杂度对比。

### 8.2 和旧版标量动态图更新的对比
对比指标：

- 距离矩阵数值误差（应为 `0`）
- assignment 是否完全一致（应为 `true`）
- kernel runtime speedup

这是实现效率对比。

### 8.3 和串行批量运行的对比
对比指标：

- total wall time
- observed speedup
- mean ms/step

这是系统吞吐对比。

### 8.4 和固定目标图的对比
对比指标：

- capture time
- mean assigned error
- final team error（作为补充）

这是算法有效性对比。

---

## 9. 目前结论能说到什么程度

当前结论是比较稳的，但要注意说法边界。

**可以稳妥地说：**

1. 当前框架已经支持一般 `m` 对 `n` 场景。
2. V-SNAC 结构有效减少了 critic 数量。
3. 向量化动态图更新在不改变分配结果的前提下显著提升了计算效率。
4. 在关闭绘图的批量 benchmark 中，并行执行显著降低了总 wall time。
5. 在多个 `m` 对 `n` 稳定场景中，动态图相对固定图能缩短捕获时间并降低平均分配误差。

**不建议说得过强：**

- 不要写“在所有 m 对 n 场景下都严格更优”
- 不要写“并行总是更快”，因为如果把大量绘图和 I/O 算进去，进程调度开销可能掩盖收益

更稳的表达是：

> 在当前稳定批量实验与 benchmark 设置下，所提出的通用化与计算优化策略能够同时提升框架可扩展性与运行效率，并保持动态图追逃机制在多组 `m` 对 `n` 场景下的有效性。

---

## 10. 下一步建议

如果后面还要继续增强这条线，最值得补的两项是：

1. 更中性的场景生成模式
   - 不只用“有利于交换”的 shifted 初始 assignment
   - 增加 neutral / random 模式

2. 多 seed 统计
   - 对每个 `m,n` 场景重复多次
   - 报均值和标准差

这样就能把当前“有效性证明”进一步推进到更扎实的统计结论。

---

## 11. 相关结果路径

- 稳定 `m` 对 `n` 结果：
  - [mn_batch_stable_20260324](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_batch_stable_20260324)
- 计算优化 benchmark：
  - [benchmark_mn_compute_20260324](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/benchmark_mn_compute_20260324)

这两组结果已经足够支撑当前 `m` 对 `n` 方向的论文写作。
