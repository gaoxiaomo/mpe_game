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
        "n_features_per_critic": int(final_w.shape[1]),
        "final_weight_norms": np.sqrt(np.sum(final_w ** 2, axis=1)).tolist(),
        "final_weight_vectors": np.round(final_w, 4).tolist(),
        "final_delta_per_critic": (deltas[-1].tolist() if deltas.size else []),
        "mean_residual_per_critic": np.nanmean(residuals, axis=0).tolist() if residuals.size else [],
    }


def eval_summary(result: EvalResult, label: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "label": label,
        "capture_time_s": None if result.capture_time is None else float(result.capture_time),
        "final_team_error": float(result.team_errors[-1]),
        "final_max_assigned_error": float(np.max(result.assigned_errors[-1])),
        "mean_assigned_error": float(np.mean(result.assigned_errors)),
    }
    if result.coord_metrics is not None:
        cm = result.coord_metrics
        out["d_min_mean"] = float(np.mean(cm.d_min))
        out["d_min_min"] = float(np.min(cm.d_min))
        out["angular_coverage_mean"] = float(np.mean(cm.angular_coverage))
        out["path_overlap_mean"] = float(np.mean(cm.path_overlap))
    return out


def network_summary(n_p: int, n_e: int, n_features: int) -> Dict[str, Any]:
    vsnac = n_p
    ac = 2 * (n_p + n_e)
    return {
        "v_snac_team_critics": vsnac,
        "features_per_critic": n_features,
        "total_parameters": vsnac * n_features,
        "ac_networks_estimated": ac,
    }


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
