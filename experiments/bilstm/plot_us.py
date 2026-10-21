"""Plot VCI-only models trained on three states, the six Plains states and the contiguous US.

All use the contiguous-US build (data_us.py), the 10-40 cm label and rolling origin
2021-2024. Pooled skill vs climatology and r2 per state, for all cells and for unseen 2x2
deg blocks, plus a map of the US model's skill per state. Writes figures_vci_us/*.png and
metrics_vci_us/comparison.csv.

    python plot_us.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_us import us_states
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import PathPatch
from shared_split_us import PLAINS, THREE_STATES
from train import INK, INK_2, SURFACE, style

OUT = Path(__file__).parent
FIG = OUT / "figures_vci_us"
METRICS = OUT / "metrics_vci_us"
# VCI-only models on an ordinal blue ramp (more training states -> darker); the US model
# with coordinates in the next categorical hue.
MODELS = {
    "three": ("3 states (KS, TX, CA), VCI", "#86b6ef"),
    "plains": ("6 Plains states, VCI", "#3987e5"),
    "us": ("contiguous US, VCI", "#184f95"),
    "us+coords": ("contiguous US, VCI + coordinates", "#eb6834"),
}
FOCUS = ["kansas", "texas", "california", "nebraska", "oklahoma", "south_dakota", "north_dakota"]
SKILL_MAP = LinearSegmentedColormap.from_list(
    "skill", ["#b3302f", "#e34948", "#f0efec", "#3987e5", "#1c5cab"]
)


def label(state):
    return state.replace("_", " ").title()


def load():
    paths = [p for p in sorted(METRICS.glob("*.csv")) if p.stem.split("_")[0] in MODELS]
    paths = [p for p in paths if "effective" not in p.stem]
    data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return data.assign(model=data.region + np.where(data.inputs == "coords", "+coords", ""))


def pooled(frame):
    sse, clim = (frame.mse * frame.n).sum(), (frame.climatology_mse * frame.n).sum()
    return pd.Series(
        {
            "skill": 1 - sse / clim,
            "r2": frame.r2.mean(),
            "worst_year_skill": (1 - frame.mse / frame.climatology_mse).min(),
            "n": frame.n.sum(),
            "years": frame.year.nunique(),
        }
    )


def summary(data):
    per_state = data.groupby(["model", "run", "state"]).apply(pooled, include_groups=False)
    groups = {"three states": THREE_STATES, "plains states": PLAINS, "all trained states": None}
    extra = []
    for name, states in groups.items():
        sub = data if states is None else data[data.state.isin(states)]
        g = sub.groupby(["model", "run"]).apply(pooled, include_groups=False)
        extra.append(g.assign(state=name).set_index("state", append=True))
    return pd.concat([per_state, *extra]).reset_index()


def bars(table):
    groups = [*FOCUS, "three states", "plains states"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.8), facecolor=SURFACE)
    width = 0.2
    for col, (run, title) in enumerate(
        (("test", "all cells"), ("spatial_unseen", "unseen 2x2° blocks"))
    ):
        for row, (metric, name) in enumerate((("skill", "skill vs climatology"), ("r2", "r²"))):
            ax = axes[row, col]
            for j, (model, (_, color)) in enumerate(MODELS.items()):
                sub = table[(table.model == model) & (table.run == run)].set_index("state")
                for k, g in enumerate(groups):
                    if g not in sub.index or (model == "three" and g == "plains states"):
                        continue
                    v, x = sub.loc[g, metric], k + (j - 1.5) * width
                    ax.bar(x, v, width - 0.03, color=color, edgecolor=SURFACE, linewidth=2)
                    ax.text(
                        x,
                        v + (0.006 if v >= 0 else -0.006),
                        f"{v:+.2f}" if row == 0 else f"{v:.2f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        color=INK_2,
                        fontsize=5.5,
                    )
            ax.axhline(0, color=INK_2, linewidth=1)
            ax.axvline(len(FOCUS) - 0.5, color=INK_2, linewidth=0.6, linestyle=":")
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([label(g).replace(" ", "\n", 1) for g in groups], fontsize=8)
            style(ax, f"{title}: {name}", "", name if col == 0 else "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS.values()]
    fig.legend(
        handles,
        [f"trained on {n}" for n, _ in MODELS.values()],
        frameon=False,
        labelcolor=INK,
        fontsize=9,
        ncol=2,
        loc="upper right",
    )
    fig.suptitle(
        "10-40 cm soil moisture, pooled 2021-2024: more training states",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "more_states.png", dpi=140)
    plt.close(fig)


def us_map(table):
    polygons = us_states()
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.01, right=0.88, top=0.94, bottom=0.06, wspace=0.03, hspace=0.12)
    norm = Normalize(-0.4, 0.4)
    panels = [
        (model, run, f"{inputs}, {title}")
        for model, inputs in (("us", "VCI only"), ("us+coords", "VCI + coordinates"))
        for run, title in (("test", "all cells"), ("spatial_unseen", "unseen 2x2° blocks"))
    ]
    for ax, (model, run, title) in zip(axes.ravel(), panels, strict=True):
        sub = table[(table.model == model) & (table.run == run)].set_index("state")
        if sub.empty:
            ax.axis("off")
            continue
        for state, rings in polygons.items():
            value = sub.skill.get(state, np.nan)
            for ring in rings:
                if np.isnan(value):  # no cells to score: hatched, not a colour on the scale
                    ax.add_patch(
                        PathPatch(
                            ring, facecolor=SURFACE, edgecolor="#c9c7c0", hatch="////", lw=0.5
                        )
                    )
                else:
                    ax.add_patch(
                        PathPatch(ring, facecolor=SKILL_MAP(norm(value)), edgecolor=SURFACE, lw=0.8)
                    )
        ax.set_xlim(-125, -66.5)
        ax.set_ylim(24.5, 49.5)
        ax.set_aspect(1.25, adjustable="box")
        ax.axis("off")
        ax.set_facecolor(SURFACE)
        overall = sub.loc["all trained states"]
        ax.set_title(
            f"{title}: skill (US overall {overall.skill:+.3f})",
            color=INK,
            fontsize=10,
            loc="left",
        )
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=SKILL_MAP), ax=axes, shrink=0.7, pad=0.01
    )
    bar.set_label("skill vs climatology (clipped at ±0.4)", color=INK_2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_2)
    fig.text(
        0.01,
        0.02,
        "Hatched: no kept cells (or no held-out block) in the state.",
        color=INK_2,
        fontsize=8,
    )
    fig.savefig(FIG / "us_skill_map.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    table = summary(load())
    table.to_csv(METRICS / "comparison.csv", index=False)
    bars(table)
    if (table.model == "us").any():
        us_map(table)
    view = table[table.state.isin([*FOCUS, "three states", "plains states", "all trained states"])]
    print(
        view.pivot_table(index=["run", "state"], columns="model", values="skill")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
