"""Plot VCI-only vs VCI + coordinates (vci_tactics.py / vci_coords.py results).

Pooled 2021-2024 skill vs climatology and r2, for all cells and for unseen 2x2 deg blocks.
Writes figures_vci_coords/vci_vs_coords.png and metrics_vci_coords/comparison.csv.

    python plot_coords.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train import INK, INK_2, SURFACE, style
from train_v2 import LABELS, STATE_COLORS, STATE_NAMES

OUT = Path(__file__).parent
FIG = OUT / "figures_vci_coords"
METRICS = OUT / "metrics_vci_coords"
GROUPS = ["overall", *STATE_NAMES]


def load():
    tactics = pd.read_csv(OUT / "metrics_vci_tactics/rolling_origin.csv")
    tactics = tactics[
        (tactics.weighting == "uniform")
        & (tactics.calibration == "raw")
        & tactics.label.isin(["sm_10_40", "sm_0_100"])
    ]
    tactics = tactics.assign(inputs="vci", run="test")
    frames = [tactics] + [
        pd.read_csv(p)
        for p in sorted(METRICS.glob("*_main.csv")) + sorted(METRICS.glob("*_spatial.csv"))
    ]
    cols = ["inputs", "label", "run", "year", "group", "n", "mse", "climatology_mse", "r2"]
    return pd.concat([f[cols] for f in frames], ignore_index=True)


def pooled(frame):
    sse, clim = (frame.mse * frame.n).sum(), (frame.climatology_mse * frame.n).sum()
    return pd.Series(
        {
            "skill": 1 - sse / clim,
            "r2": frame.r2.mean(),
            "worst_year_skill": (1 - frame.mse / frame.climatology_mse).min(),
            "years": frame.year.nunique(),
        }
    )


def main():
    FIG.mkdir(exist_ok=True)
    data = load()
    table = (
        data.groupby(["label", "run", "inputs", "group"])
        .apply(pooled, include_groups=False)
        .reset_index()
    )
    table.to_csv(METRICS / "comparison.csv", index=False)
    panels = [
        ("sm_10_40", "test", "10-40 cm, all cells"),
        ("sm_10_40", "spatial_unseen", "10-40 cm, unseen 2x2° blocks"),
        ("sm_0_100", "test", "0-100 cm, all cells"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
    width = 0.38
    for col, (label, run, title) in enumerate(panels):
        for row, (metric, name) in enumerate((("skill", "skill vs climatology"), ("r2", "r²"))):
            ax = axes[row, col]
            for j, inputs in enumerate(("vci", "coords")):
                sub = table[(table.label == label) & (table.run == run) & (table.inputs == inputs)]
                if sub.empty:
                    continue
                vals = [sub[sub.group == g][metric].iloc[0] for g in GROUPS]
                xs = np.arange(len(GROUPS)) + (j - 0.5) * width
                colors = (
                    ["#b5b3ad"] * len(GROUPS)
                    if inputs == "vci"
                    else [STATE_COLORS.get(g, INK) for g in GROUPS]
                )
                ax.bar(xs, vals, width - 0.04, color=colors, edgecolor=SURFACE, linewidth=2)
                for x, v in zip(xs, vals, strict=True):
                    ax.text(
                        x,
                        v + (0.005 if v >= 0 else -0.005),
                        f"{v:+.2f}" if row == 0 else f"{v:.2f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        color=INK_2,
                        fontsize=7,
                    )
            ax.axhline(0, color=INK_2, linewidth=1)
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels([LABELS.get(g, "All") for g in GROUPS])
            style(ax, f"{title}: {name}", "", name if col == 0 else "")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#b5b3ad"), plt.Rectangle((0, 0), 1, 1, color=INK)]
    fig.legend(
        handles,
        ["VCI only", "VCI + latitude/longitude (state color)"],
        frameon=False,
        labelcolor=INK,
        fontsize=9,
        ncol=2,
        loc="upper right",
    )
    fig.suptitle(
        "Adding coordinates to VCI: pooled 2021-2024 (rolling origin)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "vci_vs_coords.png", dpi=150)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
