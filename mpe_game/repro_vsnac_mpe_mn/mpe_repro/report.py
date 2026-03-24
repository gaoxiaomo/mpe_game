from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .simulator import EvalResult, TrainResult


def train_summary(train: TrainResult) -> Dict[str, Any]:
    deltas = train.delta_history
    residuals = train.residual_history
    final_w = train.weight_history[-1]
    return {
        "iterations": int(train.weight_history.shape[0] - 1),
        "final_weight_norms": np.sqrt(np.sum(final_w ** 2, axis=1)).tolist(),
        "final_weight_norm_squares": np.sum(final_w ** 2, axis=1).tolist(),
        "final_weight_vectors": np.round(final_w, 4).tolist(),
        "final_delta_per_critic": (deltas[-1].tolist() if deltas.size else []),
        "mean_residual_per_critic": np.nanmean(residuals, axis=0).tolist() if residuals.size else [],
    }


def eval_summary(result: EvalResult) -> Dict[str, Any]:
    return {
        "capture_time_s": None if result.capture_time is None else float(result.capture_time),
        "final_team_error": float(result.team_errors[-1]),
        "final_max_assigned_error": float(np.max(result.assigned_errors[-1])),
        "mean_assigned_error": float(np.mean(result.assigned_errors)),
    }


def network_summary(n_p: int, n_e: int, n_critics: int | None = None) -> Dict[str, Any]:
    vsnac = n_p if n_critics is None else n_critics
    ac = 2 * (n_p + n_e)
    return {
        "v_snac_networks": vsnac,
        "ac_networks_estimated": ac,
        "network_reduction_percent": 100.0 * (ac - vsnac) / max(ac, 1),
    }


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dump_train_log_markdown(
    path: Path,
    title: str,
    train: TrainResult,
) -> None:
    lines: list[str] = [
        f"# {title}",
        "",
        "## Iteration Summary",
        "| iter | exploration_std | delta_per_critic | residual_rms_per_critic | samples_per_critic | weight_norms |",
        "|---:|---:|---|---|---|---|",
    ]

    n_iters = train.weight_history.shape[0] - 1
    for it in range(n_iters + 1):
        w = train.weight_history[it]
        w_norms = np.sqrt(np.sum(w * w, axis=1))
        if it == 0:
            explore = "-"
            deltas = "-"
            residuals = "-"
            samples = "-"
        else:
            explore = f"{float(train.exploration_history[it - 1]):.6f}"
            deltas = np.array2string(train.delta_history[it - 1], precision=6, separator=", ")
            residuals = np.array2string(train.residual_history[it - 1], precision=6, separator=", ")
            samples = np.array2string(train.sample_history[it - 1].astype(int), separator=", ")
        norms = np.array2string(w_norms, precision=6, separator=", ")
        lines.append(f"| {it} | {explore} | `{deltas}` | `{residuals}` | `{samples}` | `{norms}` |")

    lines.extend(
        [
            "",
            "## Critic Weights Per Iteration",
        ]
    )
    for it in range(n_iters + 1):
        lines.append(f"### Iteration {it}")
        for j in range(train.weight_history.shape[1]):
            w = train.weight_history[it, j]
            w_str = np.array2string(w, precision=6, separator=", ")
            lines.append(f"- critic_{j+1}: `{w_str}`")
        lines.append("")

    dump_markdown(path, "\n".join(lines))


def dump_eval_log_markdown(
    path: Path,
    title: str,
    result: EvalResult,
) -> None:
    lines: list[str] = [
        f"# {title}",
        "",
        "## Summary",
        f"- capture_time_s: {None if result.capture_time is None else float(result.capture_time)}",
        f"- total_logged_steps: {0 if result.step_logs is None else len(result.step_logs)}",
        f"- final_team_error: {float(result.team_errors[-1]) if result.team_errors.size else None}",
        "",
        "## Per-Step Logs",
    ]

    if not result.step_logs:
        lines.append("- no step logs captured (`record_logs=False`).")
        dump_markdown(path, "\n".join(lines))
        return

    for row in result.step_logs:
        step = int(row.get("step", -1))
        t_s = float(row.get("time_s", 0.0))
        lines.append(f"### step={step}, t={t_s:.4f}s")
        lines.append("```json")
        lines.append(json.dumps(row, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    dump_markdown(path, "\n".join(lines))
