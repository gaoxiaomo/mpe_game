"""Generate multi-scenario composite figures from existing output PNGs.

Creates:
  paper_figures/fig_weight_convergence_multi.png  -- 1x3 grid (3v1, 6v3, 8v4)
  paper_figures/fig_errors_multi.png              -- 1x3 grid (3v1, 6v3, 8v4)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

matplotlib.use("Agg")

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "paper_figures"
STRUCT_DIR = BASE / "outputs" / "comm_structured"

SCENARIOS = ["3v1", "6v3", "8v4"]
LABELS = ["(a) 3v1", "(b) 6v3", "(c) 8v4"]


def make_composite(fig_name_template: str, out_name: str, figsize=(16, 4.5)):
    """Load per-scenario PNGs and arrange them in a 1x3 grid."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for idx, (scenario, label) in enumerate(zip(SCENARIOS, LABELS)):
        img_path = STRUCT_DIR / scenario / fig_name_template
        if not img_path.exists():
            raise FileNotFoundError(f"Missing: {img_path}")
        img = mpimg.imread(str(img_path))
        axes[idx].imshow(img)
        axes[idx].set_title(label, fontsize=15, fontweight="bold", pad=10)
        axes[idx].axis("off")

    fig.tight_layout(pad=1.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    make_composite("fig_weight_convergence.png", "fig_weight_convergence_multi.png")
    make_composite("fig_assigned_errors.png", "fig_errors_multi.png")
    print("Done.")


if __name__ == "__main__":
    main()
