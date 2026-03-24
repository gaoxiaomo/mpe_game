# Generalized MPE Reproduction (`m` pursuers vs `n` evaders)

This folder extends the original reproduction workflow to a more computer-oriented,
general-purpose runner that supports arbitrary pursuit-evasion sizes such as `3v1`,
`3v3`, `5v3`, and whole families of `m > n` cases.

## What is implemented

- nonlinear aircraft dynamics and constrained-input control inherited from the original reproduction
- dynamic target graph switching for multi-evader settings
- off-policy critic training with rollout sampling and least-squares policy evaluation
- generalized scenario generation for arbitrary `m` pursuers and `n` evaders
- batch execution and optional parallel processing across cases
- runtime metrics (`wall time`, `ms/step`, estimated parallel speedup)
- fair `dynamic graph` vs `fixed graph` comparison under the same trained weights and the same evaluation seed
- overall assigned residual-norm plotting and 3D GIF trajectory animation

## Key files

- generalized runner: [run_generalized.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/run_generalized.py)
- generalized scenario generator: [general_scenarios.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/mpe_repro/general_scenarios.py)
- plotting utilities: [plotting.py](/e:/毕业设计/mpe_game/repro_vsnac_mpe_mn/mpe_repro/plotting.py)

## Typical commands

Default batch (`3v1`, `3v3`, `5v3`):

```powershell
cd E:\毕业设计\mpe_game\repro_vsnac_mpe_mn
python .\run_generalized.py
```

Quick validation:

```powershell
python .\run_generalized.py --quick --parallel-workers 2
```

Single custom case:

```powershell
python .\run_generalized.py --n-pursuers 5 --n-evaders 3
```

Explicit case list:

```powershell
python .\run_generalized.py --case 3x1 3x3 5x3 6x4 --parallel-workers 2
```

Sweep a family of `m > n` cases:

```powershell
python .\run_generalized.py --sweep-pursuers 4:8 --sweep-evaders 1:4 --only-m-gt-n --parallel-workers 0
```

Notes:

- `--parallel-workers 0` means auto-select a worker count from CPU cores and job count.
- Range syntax accepts `a:b`; list syntax accepts `a,b,c`.

## Output structure

Each run writes to `outputs/generalized_<timestamp>/`.

Root files:

- `batch_summary.json`
- `REPORT.md`
- `fig_runtime_ms_per_step.png`

Each case folder (for example `5v3`) contains:

- `summary.json`
- `REPORT.md`
- `fig_trajectory_xy.png`
- `fig_trajectory_3d.gif`
- `fig_trajectory_multiview.png`
- `fig_assigned_errors.png`
- `fig_assigned_residual_norm.png`
- `fig_control_inputs.png`
- `fig_weight_convergence.png`

Additional files for `n > 1`:

- `fig_team_error_compare.png`
- `fig_old_new_errors.png`
- `fig_assignment_timeline.png`

## Evaluation logic

- `3v1`: cooperative multi-pursuer capture without graph switching
- `n > 1`:
  - train once
  - evaluate the trained weights under `dynamic graph`
  - evaluate the same weights again under `fixed graph`
  - keep the same evaluation seed and the same simulation horizon
  - compare capture time, mean assigned error, final horizon `Eteam`, switching behavior, and runtime

## Current role of this folder

This folder is used to validate:

- generalization from benchmark settings (`3v1`, `3v3`) to larger `m > n` scenarios
- computer-oriented batch execution and parallel processing
- scalable visualization and summary generation for multi-case experiments
