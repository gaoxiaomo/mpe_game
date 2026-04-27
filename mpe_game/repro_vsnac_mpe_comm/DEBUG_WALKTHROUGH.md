# Debug Walkthrough

## 最短入口

- 单 case 调试入口：
  [run_debug_demo.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/run_debug_demo.py)
- 全量实验入口：
  [run_comm.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/run_comm.py)

如果你现在想单步看系统，优先从 `run_debug_demo.py` 开始，不要一上来跑 `run_comm.py` 的整批流程。

## 建议阅读顺序

1. [run_debug_demo.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/run_debug_demo.py)
2. [mpe_repro/simulator.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/simulator.py)
3. [mpe_repro/controller.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/controller.py)
4. [mpe_repro/comm_graph.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/comm_graph.py)
5. [mpe_repro/offpolicy_ls.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/offpolicy_ls.py)
6. [mpe_repro/plotting.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/plotting.py)

## 最值得打断点的位置

- [run_debug_demo.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/run_debug_demo.py)
  先在 `run_debug_demo(...)` 里 `sim.train_policy(...)` 前后各打一处，确认场景和参数。
- [mpe_repro/simulator.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/simulator.py)
  在 `_step(...)` 里看一次完整链路：assignment -> adjacency -> control -> rk4。
- [mpe_repro/controller.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/controller.py)
  在 `coordination_potential_and_gradient(...)` 里看平滑因子、pair weight、解析梯度。
- [mpe_repro/controller.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/controller.py)
  在 `policy(...)` 里看 `grad_p`、`total_grad`、`u_p`、`u_e_virtual`。
- [mpe_repro/offpolicy_ls.py](E:/mpe_game/mpe_game/repro_vsnac_mpe_comm/mpe_repro/offpolicy_ls.py)
  在 `add_sample(...)` 和 `solve(...)` 里看 Bellman 行和 LS 求解结果。

## 推荐观察变量

- `assigned`
- `A_p`
- `delta_matrix[0, 1, :3]`
- `x_err[0]`
- `coordination_grads[0]`
- `value_terms_t`
- `value_terms_tp1`
- `u_p[0]`
- `step.stage_costs`

## 推荐命令

```powershell
python run_debug_demo.py --case 3v1 --quick --output outputs\debug_demo_3v1
```

```powershell
python run_debug_demo.py --case 6v3 --quick --output outputs\debug_demo_6v3
```

## Demo 输出

每次 `run_debug_demo.py` 都会生成：

- `summary.json`
- `step_logs.json`
- `TRAIN_LOG.md`
- `EVAL_LOG.md`
- `fig_trajectory_multiview.png`
- `fig_trajectory_3d.gif`
- `fig_assigned_errors.png`
- `fig_d_min.png`

其中 `fig_trajectory_3d.gif` 就是最适合现场演示的动态文件。
