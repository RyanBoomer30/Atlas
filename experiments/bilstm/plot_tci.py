"""Plot the West-only models and the VCI + TCI models (vci_us.py --region west/east/..., --inputs tci).

Each region is scored on its own states, pooled over 2021-2024 (10-40 cm label). Writes
figures_vci_us/west_only.png, tci_vs_vci.png, us_tci_map.png and metrics_vci_us/regions.csv.

    python plot_tci.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_us import us_states
from matplotlib.colors import Normalize
from matplotlib.patches import PathPatch
from plot_us import FIG, METRICS, SKILL_MAP, label, load, pooled
from shared_split_us import EAST, THREE_STATES, WEST
from train import INK, INK_2, SURFACE, style

REGIONS = {"us": None, "west": WEST, "east": EAST, "three": THREE_STATES}
REGION_LABELS = {
    "us": "US model,\nall 38 states",
    "west": "West model,\n16 states",
    "east": "East model,\n22 states",
    "three": "KS+TX+CA model,\n3 states",
}
INPUTS = {"vci": ("VCI only", "#2a78d6"), "tci": ("VCI + TCI", "#eb6834")}
WEST_MODELS = {
    "us": ("US model, VCI", "#86b6ef"),
    "west": ("West model, VCI", "#2a78d6"),
    "west+coords": ("West model, VCI + coordinates", "#184f95"),
    "west+tci": ("West model, VCI + TCI", "#eb6834"),
}
WEST_GROUPS = ["west states", "kansas", "texas", "california", "nebraska", "south_dakota",
               "colorado", "montana"]  # fmt: skip
RUNS = (("test", "all cells"), ("spatial_unseen", "unseen 2x2° blocks"))


def region_table(data):
    """Per model: pooled over the region's own states, per state, and West / East / 3 states."""
    data = data.assign(model=data.region + np.where(data.inputs == "vci", "", "+" + data.inputs))
    per_state = data.groupby(["region", "inputs", "model", "run", "state"]).apply(
        pooled, include_groups=False
    )
    groups = {"own states": None, "west states": WEST, "east states": EAST,
              "three states": THREE_STATES}  # fmt: skip
    extra = []
    for name, states in groups.items():
        sub = data if states is None else data[data.state.isin(states)]
        g = sub.groupby(["region", "inputs", "model", "run"]).apply(pooled, include_groups=False)
        extra.append(g.assign(state=name).set_index("state", append=True))
    return pd.concat([per_state, *extra]).reset_index()


def labelled_bar(ax, x, v, width, color, row):
    ax.bar(x, v, width, color=color, edgecolor=SURFACE, linewidth=2)
    ax.text(
        x,
        v + (0.006 if v >= 0 else -0.006),
        f"{v:+.2f}" if row == 0 else f"{v:.2f}",
        ha="center",
        va="bottom" if v >= 0 else "top",
        color=INK_2,
        fontsize=6.5,
    )


def finish(fig, axes, handles, names, title, path, ncol):
    fig.legend(
        handles, names, frameon=False, labelcolor=INK, fontsize=9, ncol=ncol, loc="upper right"
    )
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / path, dpi=140)
    plt.close(fig)


def tci_bars(table):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), facecolor=SURFACE)
    width = 0.36
    for col, (run, title) in enumerate(RUNS):
        for row, (metric, name) in enumerate((("skill", "skill vs climatology"), ("r2", "r²"))):
            ax = axes[row, col]
            for k, region in enumerate(REGIONS):
                for j, (inputs, (_, color)) in enumerate(INPUTS.items()):
                    sel = table[
                        (table.region == region)
                        & (table.inputs == inputs)
                        & (table.run == run)
                        & (table.state == "own states")
                    ]
                    if len(sel):
                        x = k + (j - 0.5) * width
                        labelled_bar(ax, x, sel[metric].iloc[0], width - 0.04, color, row)
            ax.axhline(0, color=INK_2, linewidth=1)
            ax.set_xticks(range(len(REGIONS)))
            ax.set_xticklabels(REGION_LABELS.values(), fontsize=8.5)
            style(ax, f"{title}: {name}", "", name if col == 0 else "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in INPUTS.values()]
    finish(
        fig,
        axes,
        handles,
        [n for n, _ in INPUTS.values()],
        "Adding land-surface temperature (TCI) to VCI: 10-40 cm soil moisture, pooled 2021-2024",
        "tci_vs_vci.png",
        2,
    )


def west_bars(table):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.6), facecolor=SURFACE)
    width = 0.2
    for col, (run, title) in enumerate(RUNS):
        for row, (metric, name) in enumerate((("skill", "skill vs climatology"), ("r2", "r²"))):
            ax = axes[row, col]
            for j, (model, (_, color)) in enumerate(WEST_MODELS.items()):
                sub = table[(table.model == model) & (table.run == run)].set_index("state")
                for k, g in enumerate(WEST_GROUPS):
                    if g in sub.index:
                        labelled_bar(
                            ax, k + (j - 1.5) * width, sub.loc[g, metric], 0.17, color, row
                        )
            ax.axhline(0, color=INK_2, linewidth=1)
            ax.axvline(0.5, color=INK_2, linewidth=0.6, linestyle=":")
            ax.set_xticks(range(len(WEST_GROUPS)))
            ax.set_xticklabels([label(g).replace(" ", "\n", 1) for g in WEST_GROUPS], fontsize=8)
            style(ax, f"{title}: {name}", "", name if col == 0 else "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in WEST_MODELS.values()]
    finish(
        fig,
        axes,
        handles,
        [n for n, _ in WEST_MODELS.values()],
        "Western states (west of 98°W): US model vs West-only models, pooled 2021-2024",
        "west_only.png",
        2,
    )


def tci_map(table):
    polygons = us_states()
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.01, right=0.88, top=0.94, bottom=0.06, wspace=0.03, hspace=0.12)
    norm = Normalize(-0.4, 0.4)
    us = table[table.region == "us"]
    for row, (run, title) in enumerate(RUNS):
        vci = us[(us.inputs == "vci") & (us.run == run)].set_index("state").skill
        tci = us[(us.inputs == "tci") & (us.run == run)].set_index("state").skill
        panels = ((tci, f"VCI + TCI, {title}: skill"), (tci - vci, f"{title}: gain from TCI"))
        for ax, (values, name) in zip(axes[row], panels, strict=True):
            for state, rings in polygons.items():
                v = values.get(state, np.nan)
                for ring in rings:
                    if np.isnan(v):
                        patch = PathPatch(
                            ring, facecolor=SURFACE, edgecolor="#c9c7c0", hatch="////", lw=0.5
                        )
                    else:
                        patch = PathPatch(ring, facecolor=SKILL_MAP(norm(v)), edgecolor=SURFACE)
                    ax.add_patch(patch)
            ax.set_xlim(-125, -66.5)
            ax.set_ylim(24.5, 49.5)
            ax.set_aspect(1.25, adjustable="box")
            ax.axis("off")
            overall = values.get("own states", np.nan)
            ax.set_title(f"{name} (US {overall:+.3f})", color=INK, fontsize=10, loc="left")
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=SKILL_MAP), ax=axes, shrink=0.7, pad=0.01
    )
    bar.set_label("skill, or skill gain (clipped at ±0.4)", color=INK_2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_2)
    fig.text(0.01, 0.02, "Hatched: no kept cells (or no held-out block).", color=INK_2, fontsize=8)
    fig.savefig(FIG / "us_tci_map.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    table = region_table(load())
    table.to_csv(METRICS / "regions.csv", index=False)
    west_bars(table)
    tci_bars(table)
    if ((table.region == "us") & (table.inputs == "tci")).any():
        tci_map(table)
    view = table[table.state.isin(["own states", "west states", "east states", "three states"])]
    print(view.pivot_table(index=["run", "state"], columns="model", values="skill").round(3))


if __name__ == "__main__":
    main()
