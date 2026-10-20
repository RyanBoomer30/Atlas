"""Plot single-state vs three-state VCI-only models (vci_per_state.py vs vci_tactics.py).

Both use the 10-40 cm label, uniform weights, raw output, rolling origin 2021-2024 and the
same number of training draws. Writes figures_vci_per_state/per_state_vs_pooled.png and
metrics_vci_per_state/comparison.csv.

    python plot_per_state.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train import GRID, INK, INK_2, SURFACE, style
from train_v2 import LABELS, STATE_COLORS, STATE_NAMES

OUT = __import__("pathlib").Path(__file__).parent
FIG = OUT / "figures_vci_per_state"
SINGLE = OUT / "metrics_vci_per_state"
POOLED = OUT / "metrics_vci_tactics/rolling_origin.csv"
YEARS = (2021, 2022, 2023, 2024)


def load():
    single = pd.concat(pd.read_csv(SINGLE / f"{s}.csv") for s in STATE_NAMES)
    single = single.assign(model="single-state")
    pooled = pd.read_csv(POOLED)
    pooled = pooled[
        (pooled.label == "sm_10_40")
        & (pooled.weighting == "uniform")
        & (pooled.calibration == "raw")
        & (pooled.group.isin(STATE_NAMES))
    ]
    pooled = pooled.rename(columns={"group": "state"}).assign(model="three-state")
    cols = [
        "model",
        "state",
        "year",
        "n_test",
        "mse",
        "climatology_mse",
        "persistence_mse",
        "skill_vs_climatology",
        "R2",
        "r2",
    ]
    pooled = pooled.rename(columns={"n": "n_test"})
    return pd.concat([single[cols], pooled[cols]], ignore_index=True)


def pooled_row(frame):
    sse, clim = (frame.mse * frame.n_test).sum(), (frame.climatology_mse * frame.n_test).sum()
    return pd.Series(
        {
            "skill_vs_climatology": 1 - sse / clim,
            "r2": frame.r2.mean(),
            "mse": sse / frame.n_test.sum(),
        }
    )


def main():
    FIG.mkdir(exist_ok=True)
    data = load()
    summary = (
        data.groupby(["model", "state"])
        .apply(pooled_row, include_groups=False)
        .reset_index()
        .assign(year="2021-24")
    )
    table = pd.concat([data.assign(year=data.year.astype(str)), summary], ignore_index=True)
    table.to_csv(SINGLE / "comparison.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURFACE, sharex=True)
    groups = [str(y) for y in YEARS] + ["2021-24"]
    width = 0.38
    for col, state in enumerate(STATE_NAMES):
        for row, (metric, label) in enumerate(
            (("skill_vs_climatology", "skill vs climatology"), ("r2", "r² (timing)"))
        ):
            ax = axes[row, col]
            for j, model in enumerate(("three-state", "single-state")):
                vals = [
                    table[(table.model == model) & (table.state == state) & (table.year == g)][
                        metric
                    ].iloc[0]
                    for g in groups
                ]
                xs = np.arange(len(groups)) + (j - 0.5) * width
                color = "#b5b3ad" if model == "three-state" else STATE_COLORS[state]
                ax.bar(
                    xs,
                    vals,
                    width - 0.04,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=2,
                    label=f"{model} model",
                )
                for x, v in zip(xs, vals, strict=True):
                    ax.text(
                        x,
                        v + (0.01 if v >= 0 else -0.01),
                        f"{v:+.2f}" if row == 0 else f"{v:.2f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        color=INK_2,
                        fontsize=7,
                    )
            ax.axhline(0, color=INK_2, linewidth=1)
            ax.axvline(len(groups) - 1.5, color=GRID, linewidth=1)
            style(
                ax,
                f"{LABELS[state]}: {label}",
                "test year" if row == 1 else "",
                label if col == 0 else "",
            )
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups)
            if row == 0:
                ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc="lower left")
        lo = min(table[table.state.isin(STATE_NAMES)].skill_vs_climatology.min(), -0.05)
        hi = max(table[table.state.isin(STATE_NAMES)].skill_vs_climatology.max(), 0.05)
        axes[0, col].set_ylim(lo - 0.06, hi + 0.06)
        axes[1, col].set_ylim(0, max(table.r2.max(), 0.05) + 0.04)
    fig.suptitle(
        "VCI-only, 10-40 cm soil moisture: one model per state vs one model for all "
        "three (same training steps)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "per_state_vs_pooled.png", dpi=150)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
