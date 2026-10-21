"""Plot regional VCI-only models vs the three-state models (VCI only, VCI + coordinates).

All use the 10-40 cm label and rolling origin 2021-2024. Pooled skill vs climatology and r2
per state, for all cells and for unseen 2x2 deg blocks. "All states" pools the per-state
errors. Writes figures_vci_regional/regional_vs_pooled.png and metrics_vci_regional/comparison.csv.

    python plot_regional.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train import INK, INK_2, SURFACE, style
from train_v2 import LABELS, STATE_COLORS, STATE_NAMES

OUT = Path(__file__).parent
FIG = OUT / "figures_vci_regional"
METRICS = OUT / "metrics_vci_regional"
MODELS = {
    "three-state, VCI only": "#d6d4ce",
    "three-state, VCI + coordinates": "#8f8d87",
    "regional, VCI only": None,  # state color
}
COLS = ["model", "run", "year", "state", "n", "mse", "climatology_mse", "r2"]


def load():
    frames = []
    tactics = pd.read_csv(OUT / "metrics_vci_tactics/rolling_origin.csv")
    tactics = tactics[
        (tactics.label == "sm_10_40")
        & (tactics.weighting == "uniform")
        & (tactics.calibration == "raw")
    ]
    frames.append(tactics.assign(model="three-state, VCI only", run="test"))
    for name, model in (
        ("coords_sm_10_40_main", "three-state, VCI + coordinates"),
        ("coords_sm_10_40_spatial", "three-state, VCI + coordinates"),
        ("vci_sm_10_40_spatial", "three-state, VCI only"),
    ):
        frames.append(pd.read_csv(OUT / f"metrics_vci_coords/{name}.csv").assign(model=model))
    for path in sorted(METRICS.glob("*_main.csv")) + sorted(METRICS.glob("*_spatial.csv")):
        frames.append(pd.read_csv(path).assign(model="regional, VCI only"))
    data = pd.concat(frames, ignore_index=True).rename(columns={"group": "state"})
    return data[data.state.isin(STATE_NAMES)][COLS]


def pooled(frame):
    sse, clim = (frame.mse * frame.n).sum(), (frame.climatology_mse * frame.n).sum()
    return pd.Series(
        {
            "skill": 1 - sse / clim,
            "r2": frame.r2.mean(),
            "worst_year_skill": (1 - frame.mse / frame.climatology_mse).min(),
        }
    )


def main():
    FIG.mkdir(exist_ok=True)
    data = load()
    per_state = data.groupby(["model", "run", "state"]).apply(pooled, include_groups=False)
    overall = data.groupby(["model", "run"]).apply(pooled, include_groups=False)
    overall = overall.assign(state="overall").set_index("state", append=True)
    table = pd.concat([per_state, overall]).reset_index()
    table.to_csv(METRICS / "comparison.csv", index=False)

    groups = ["overall", *STATE_NAMES]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), facecolor=SURFACE)
    width = 0.27
    for col, (run, title) in enumerate(
        (("test", "all cells"), ("spatial_unseen", "unseen 2x2° blocks"))
    ):
        for row, (metric, name) in enumerate((("skill", "skill vs climatology"), ("r2", "r²"))):
            ax = axes[row, col]
            for j, (model, color) in enumerate(MODELS.items()):
                sub = table[(table.model == model) & (table.run == run)].set_index("state")
                vals = [sub.loc[g, metric] for g in groups]
                xs = np.arange(len(groups)) + (j - 1) * width
                colors = [color or STATE_COLORS.get(g, INK) for g in groups]
                ax.bar(xs, vals, width - 0.03, color=colors, edgecolor=SURFACE, linewidth=2)
                for x, v in zip(xs, vals, strict=True):
                    ax.text(
                        x,
                        v + (0.006 if v >= 0 else -0.006),
                        f"{v:+.2f}" if row == 0 else f"{v:.2f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        color=INK_2,
                        fontsize=7,
                    )
            ax.axhline(0, color=INK_2, linewidth=1)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([LABELS.get(g, "All states") for g in groups])
            style(ax, f"{title}: {name}", "", name if col == 0 else "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c or INK) for c in MODELS.values()]
    fig.legend(
        handles,
        [
            "three-state, VCI only",
            "three-state, VCI + coordinates",
            "regional (California | Kansas + Texas), VCI only — state color",
        ],
        frameon=False,
        labelcolor=INK,
        fontsize=8.5,
        ncol=3,
        loc="upper right",
    )
    fig.suptitle(
        "10-40 cm soil moisture, pooled 2021-2024", x=0.01, ha="left", color=INK, fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "regional_vs_pooled.png", dpi=150)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
