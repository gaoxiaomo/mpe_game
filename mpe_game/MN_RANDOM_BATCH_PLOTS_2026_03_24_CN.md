# 随机多组 `m` 对 `n` 结果说明（带图版本）

## 输出路径

- 总目录：[`E:\毕业设计\mpe_game\repro_vsnac_mpe_mn\outputs\mn_random_batch_plots_20260324`](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_random_batch_plots_20260324)
- 总报告：[`E:\毕业设计\mpe_game\repro_vsnac_mpe_mn\outputs\mn_random_batch_plots_20260324\REPORT.md`](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_random_batch_plots_20260324/REPORT.md)
- 总汇总：[`E:\毕业设计\mpe_game\repro_vsnac_mpe_mn\outputs\mn_random_batch_plots_20260324\batch_summary.json`](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_random_batch_plots_20260324/batch_summary.json)

目前 `E:\毕业设计\mpe_game\repro_vsnac_mpe_mn\outputs` 下只保留这一组最新随机批量结果。

## 本次设置

- 使用当前加速后的 `m` 对 `n` 版本
- `assignment_mode = random`
- `layout_mode = random`
- 随机初值与随机分配同时启用
- 相比上一版，进一步拉大了 pursuer 与 evader 的初始空间间距
- 对随机几何生成增加了**最小间距约束**
- 保留绘图输出，不使用 `--skip-plots`

## 运行的场景

- `3v1`
- `4v2`
- `4v3`
- `5v2`
- `5v3`
- `5v4`
- `6v2`
- `6v3`
- `6v4`
- `7v3`
- `7v4`
- `8v4`

## 运行效率

- 并行 worker：`12`
- 总 wall time：`97.026 s`
- 各 case wall time 求和：`448.539 s`
- 观测并行加速：`4.623x`

由于本次保留了绘图与 GIF 导出，因此总 wall time 比纯仿真 benchmark 更高，这是正常现象。

## 主要结果

| case | capture(dynamic) | mean assigned(dynamic) | final Eteam(dynamic) | capture(fixed) | mean assigned(fixed) | final Eteam(fixed) | switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 64.65 | 2600.111 | 257.168 | - | - | - | 0 |
| 4v2 | 64.05 | 2172.787 | 247.380 | 72.65 | 2304.429 | 258.428 | 1 |
| 4v3 | 53.70 | 2007.107 | 249.640 | 52.90 | 2256.794 | 260.342 | 1 |
| 5v2 | 70.40 | 2022.555 | 300.938 | 59.65 | 2045.654 | 297.997 | 1 |
| 5v3 | 69.75 | 2215.064 | 331.205 | 65.75 | 2188.831 | 321.312 | 3 |
| 5v4 | 72.25 | 1912.130 | 345.181 | 77.85 | 2185.374 | 346.409 | 1 |
| 6v2 | 73.15 | 2129.651 | 377.146 | 80.30 | 2203.573 | 378.550 | 1 |
| 6v3 | 49.90 | 2047.835 | 363.129 | 49.65 | 2090.312 | 368.068 | 1 |
| 6v4 | 63.60 | 2120.451 | 410.170 | 60.80 | 2154.186 | 413.738 | 1 |
| 7v3 | 81.95 | 2668.017 | 511.708 | 78.70 | 2668.416 | 493.622 | 2 |
| 7v4 | - | 3144.058 | 2332.772 | 72.90 | 2619.185 | 505.626 | 4 |
| 8v4 | 77.20 | 2135.017 | 581.266 | 83.40 | 2291.853 | 623.587 | 2 |

## 如何理解这组结果

1. 这组结果比之前更“中性”
   
   因为起点和分配都是随机的，而且初始几何被主动拉开，所以它不再是容易触发交换的特例场景。

2. 动态图仍然在不少场景里有效
   
   例如：
   - `4v2`：动态捕获更快，平均分配误差更低
   - `5v4`：动态捕获更快，平均分配误差更低
   - `6v2`：动态捕获更快，平均分配误差更低
   - `8v4`：动态捕获更快，平均分配误差更低

3. 但随机场景下收益已经是明显的“场景相关”问题
   
   例如：
   - `4v3`、`6v3` 的 capture time 差距很小
   - `5v2`、`5v3`、`6v4`、`7v3` 中固定图在当前单 seed 下更快
   - `7v4` 中动态图在当前时域内未完成捕获

因此，这组结果更适合支撑下面这个结论：

> 当前加速后的 `m` 对 `n` 框架在随机初值与随机分配下仍能稳定运行；动态图机制在不少随机场景中仍然有效，但其收益具有显著的场景依赖性。

## 这组结果里有哪些图

每个 case 目录里都包含：

- 轨迹图：`fig_trajectory_xy.png`、`fig_trajectory_multiview.png`
- 3D 动图：`fig_trajectory_3d.gif`
- 总体残差图：`fig_assigned_residual_norm.png`
- 团队误差图：`fig_team_error_compare.png`
- 控制输入图：`fig_control_inputs.png`
- 权重收敛图：`fig_weight_convergence.png`
- 分配时间线：`fig_assignment_timeline.png`

例如 `5v3` 的图在：

- [`E:\毕业设计\mpe_game\repro_vsnac_mpe_mn\outputs\mn_random_batch_plots_20260324\5v3`](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/outputs/mn_random_batch_plots_20260324/5v3)

## 当前建议

如果后面要把这条线写进论文，建议主打：

1. 通用 `m` 对 `n` 扩展能力
2. 向量化动态图更新
3. 并行批量验证能力
4. 在随机场景下的可行性与可扩展性

而不要把结论写成“动态图在所有随机 `m > n` 场景下都显著优于固定图”。
