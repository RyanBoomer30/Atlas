"""Train VCI -> FLDAS root-zone soil moisture (see SPLIT_vci.md, RESULTS_vci.md).

Two variants, both with the train_v2_delta.py recipe (hidden 64, AdamW weight decay
1e-2, early stopping, effective-sample-size state shares, seeds 42-44, spatial holdout):

- vci_only: input VCI + month sin/cos; predicts the soil-moisture anomaly at t+1.
- vci_sm_change: adds the soil-moisture anomaly at t-11..t and predicts its change
  (forecast = anomaly at t + output), the setup that beat persistence for SPEI.

Baselines: climatology (anomaly 0) and persistence (anomaly at t).

    python train_vci.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import train_v2_delta as D
from data_vci import SM_HISTORY_FEATURE, SUMMARY_JSON, VCI_FEATURES, load_samples
from train import GRID, INK, INK_2, SURFACE, style
from train_v2 import LABELS, STATE_COLORS, STATE_NAMES, equivalence, scores
from train_v2 import spatial_holdout as holdout_blocks

OUT = D.OUT
FIG = OUT / "figures_vci"
METRICS = OUT / "metrics_vci"
SEEDS = (42, 43, 44)
WEIGHT_DECAY = 1e-2
VARIANTS = {
    "vci_only": {"sm_history": False, "change": False, "label": "VCI only"},
    "vci_sm_change": {
        "sm_history": True,
        "change": True,
        "label": "VCI + SM history, change target",
    },
}
VARIANT_COLORS = {"vci_only": "#9c9a94", "vci_sm_change": INK}


def baseline_rows(samples, mask, run):
    y, persist, states = samples["y"][mask], samples["persistence"][mask], samples["state"][mask]
    rows = []
    for group in ["overall", *STATE_NAMES]:
        sel = np.ones(len(y), bool) if group == "overall" else states == STATE_NAMES.index(group)
        for name, guess in (("climatology", np.zeros(sel.sum())), ("persistence", persist[sel])):
            rows.append(
                {"variant": name, "run": run, "seed": 0, "group": group, "n": int(sel.sum())}
                | scores(guess, y[sel])
            )
    return rows


def run_variant(key, spec, shares):
    samples = load_samples(sm_history=spec["sm_history"])
    D.INPUTS = VCI_FEATURES + ([SM_HISTORY_FEATURE] if spec["sm_history"] else [])
    assert samples["x"].shape[2] == len(D.INPUTS)
    # A level target is the change target with persistence set to zero.
    train_samples = (
        samples
        if spec["change"]
        else samples | {"persistence": np.zeros_like(samples["persistence"])}
    )
    test = samples["split"] == "test"
    held, _, _ = holdout_blocks(samples)
    rows, gaps, runs, preds = [], {}, {}, []
    for seed in SEEDS:
        run = D.fit(train_samples, shares, f"vci_{key}_seed{seed}", seed, WEIGHT_DECAY)
        pred = D.forecast(run["model"], train_samples, test)
        preds.append(pred)
        for r in D.evaluate(pred, samples, test, seed, "test"):
            if r["model"] == "bilstm_delta":
                rows.append({"variant": key} | {k: v for k, v in r.items() if k != "model"})
        gaps[f"test_seed{seed}"] = D.bootstrap_gap(pred, samples, test)
        spatial = D.fit(
            train_samples, shares, f"vci_{key}_seed{seed}_spatial", seed, WEIGHT_DECAY, held
        )
        unseen = test & held
        upred = D.forecast(spatial["model"], train_samples, unseen)
        for r in D.evaluate(upred, samples, unseen, seed, "spatial_unseen"):
            if r["model"] == "bilstm_delta":
                rows.append({"variant": key} | {k: v for k, v in r.items() if k != "model"})
        runs[seed] = {k: run[k] for k in ("best_epoch", "epochs_run", "best_val", "seconds")}
        if seed == SEEDS[0]:
            torch.save(run["model"].state_dict(), OUT / f"scratch_bilstm_{key}_seed{seed}.pt")
            runs["equivalence_pass"] = equivalence(run["model"])[1]
        print(key, seed, runs[seed], flush=True)
    rows += baseline_rows(samples, test, "test")
    rows += baseline_rows(samples, test & held, "spatial_unseen")
    return rows, gaps, runs, np.mean(preds, axis=0), samples


def figure_skill(table):
    test = table[table.run == "test"]
    base = test[test.variant == "persistence"].set_index("group").mse
    groups = ["overall", *STATE_NAMES]
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=SURFACE)
    width = 0.27
    for j, key in enumerate(["climatology", *VARIANTS]):
        for i, g in enumerate(groups):
            rel = test[(test.variant == key) & (test.group == g)].mse.to_numpy() / base[g] - 1
            x = i + (j - 1) * width
            color = (
                "#d6d4ce"
                if key == "climatology"
                else (STATE_COLORS.get(g, INK) if key == "vci_sm_change" else "#9c9a94")
            )
            ax.bar(x, rel.mean(), width - 0.04, color=color, edgecolor=SURFACE, linewidth=2)
            if len(rel) > 1:
                ax.plot([x] * len(rel), rel, "o", color=INK, markersize=3, zorder=3)
            ax.text(
                x,
                rel.mean() + (0.03 if rel.mean() >= 0 else -0.03),
                f"{rel.mean():+.0%}",
                ha="center",
                va="bottom" if rel.mean() >= 0 else "top",
                color=INK_2,
                fontsize=7,
            )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#d6d4ce", "#9c9a94", INK)]
    ax.legend(
        handles,
        [
            "climatology (anomaly 0)",
            VARIANTS["vci_only"]["label"],
            VARIANTS["vci_sm_change"]["label"] + " (state color)",
        ],
        frameon=False,
        labelcolor=INK,
        fontsize=8,
        loc="upper left",
    )
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.text(len(groups) - 0.5, 0, "persistence", color=INK_2, va="bottom", ha="right", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([LABELS.get(g, "All states") for g in groups])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    style(ax, "Soil-moisture anomaly, test 2021-2024: MSE vs persistence", "", "MSE vs persistence")
    fig.tight_layout()
    fig.savefig(FIG / "skill_vs_persistence.png", dpi=160)
    plt.close(fig)


def figure_series(samples, preds):
    test = samples["split"] == "test"
    frame = pd.DataFrame(
        {
            "state": samples["state"][test],
            "month": pd.PeriodIndex(samples["target_month"][test], freq="M").to_timestamp(),
            "actual": samples["y"][test],
            "persistence": samples["persistence"][test],
            **{k: v for k, v in preds.items()},
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), facecolor=SURFACE, sharex=True)
    for i, (ax, state) in enumerate(zip(axes, STATE_NAMES, strict=True)):
        m = frame[frame.state == i].groupby("month").mean(numeric_only=True)
        ax.plot(m.index, m.actual, color=STATE_COLORS[state], linewidth=2.5, label="FLDAS (actual)")
        ax.plot(
            m.index, m.persistence, color=GRID, linewidth=1.5, linestyle="--", label="persistence"
        )
        ax.plot(m.index, m.vci_only, color="#9c9a94", linewidth=1.5, label="VCI only")
        ax.plot(m.index, m.vci_sm_change, color=INK, linewidth=1.5, label="VCI + SM history")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        style(
            ax,
            f"{LABELS[state]}: state-mean root-zone soil-moisture anomaly (z)",
            "",
            "anomaly (z)",
        )
        if i == 0:
            ax.legend(frameon=False, labelcolor=INK, fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "test_series.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)
    base = load_samples()
    n_eff = D.effective_cells(base)
    shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}
    counts = (
        pd.DataFrame({"state": np.array(STATE_NAMES)[base["state"]], "split": base["split"]})
        .query("split != 'gap'")
        .value_counts()
        .unstack(fill_value=0)[["train", "val", "test"]]
    )
    counts.to_csv(METRICS / "counts.csv")
    print(counts, "\nshares", shares)
    rows, summary, preds = [], {"effective_cells": n_eff, "state_shares": shares}, {}
    for key, spec in VARIANTS.items():
        r, gaps, runs, pred, samples = run_variant(key, spec, shares)
        rows += r
        preds[key] = pred
        summary[key] = {"runs": runs, "mse_gap_vs_persistence": gaps}
    table = pd.DataFrame(rows).drop_duplicates(["variant", "run", "seed", "group"])
    table.to_csv(METRICS / "metrics_by_seed.csv", index=False)
    agg = (
        table.groupby(["run", "variant", "group"])
        .agg(
            n=("n", "first"),
            mse_mean=("mse", "mean"),
            mse_min=("mse", "min"),
            mse_max=("mse", "max"),
            mae_mean=("mae", "mean"),
            r_mean=("r", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(METRICS / "metrics_summary.csv", index=False)
    summary["data"] = json.loads(SUMMARY_JSON.read_text())
    (METRICS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(agg.round(4).to_string(index=False))
    figure_skill(table)
    figure_series(samples, preds)
    assert all(summary[k]["runs"]["equivalence_pass"] for k in VARIANTS), "equivalence FAILED"


if __name__ == "__main__":
    main()
