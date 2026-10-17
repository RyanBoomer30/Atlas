"""Train the VCI-only daily model: VCI (360 days) -> 30-day soil-moisture anomaly.

See data_vci_daily.py for inputs and labels. Training anchors every TRAIN_STRIDE days,
validation every VAL_STRIDE days; test is scored on every day of 2021-2024 (daily
inference) and on month-end anchors. Same recipe as train_v2_delta.py (hidden 64, AdamW
weight decay 1e-2, early stopping, effective-sample-size state shares, seeds 42-44,
2x2 deg spatial holdout); the target is the level (VCI is the only input, so there is no
soil-moisture value to predict a change from).

    python train_vci_daily.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import train_v2_delta as D
from data_vci_daily import DAILY_SUMMARY, first_anchor, load_daily, samples_for
from train import GRID, INK, INK_2, SURFACE, style
from train_v2 import LABELS, STATE_COLORS, STATE_NAMES, equivalence, scores
from train_v2 import spatial_holdout as holdout_blocks

OUT = D.OUT
FIG = OUT / "figures_vci_daily"
METRICS = OUT / "metrics_vci_daily"
SEEDS = (42, 43, 44)
WEIGHT_DECAY = 1e-2
TRAIN_STRIDE = VAL_STRIDE = 5


def select(samples, mask):
    return {k: v[mask] for k, v in samples.items()}


def level(samples):
    """Train on the anomaly itself: the change target with persistence set to zero."""
    return samples | {"persistence": np.zeros_like(samples["persistence"])}


def evaluate(pred, samples, run, seed):
    y, persist, states = samples["y"], samples["persistence"], samples["state"]
    rows = []
    for group in ["overall", *STATE_NAMES]:
        sel = np.ones(len(y), bool) if group == "overall" else states == STATE_NAMES.index(group)
        for model, guess in (
            ("vci_daily", pred[sel]),
            ("persistence", persist[sel]),
            ("climatology", np.zeros(sel.sum())),
        ):
            if model != "vci_daily" and seed != SEEDS[0]:
                continue
            rows.append(
                {
                    "run": run,
                    "seed": seed if model == "vci_daily" else 0,
                    "model": model,
                    "group": group,
                    "n": int(sel.sum()),
                }
                | scores(guess, y[sel])
            )
    return rows


def by_month(samples):
    """Bootstrap unit: the anchor's calendar month (days within a month move together)."""
    return samples | {"target_month": np.char.ljust(samples["target_month"].astype("U7"), 7)}


def figure_series(series):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.4), facecolor=SURFACE, sharex=True)
    for ax, state in zip(axes, STATE_NAMES, strict=True):
        m = series[series.state == state]
        when = pd.DatetimeIndex(m.anchor)
        ax.plot(when, m.actual, color=STATE_COLORS[state], linewidth=2.2)
        ax.plot(when, m.persistence, color=GRID, linewidth=1.4, linestyle="--")
        ax.plot(when, m.prediction, color=INK, linewidth=1.3)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        style(
            ax,
            f"{LABELS[state]}: forecast issued each day for the next 30 days",
            "",
            "SM anomaly (z)",
        )
    handles = [
        plt.Line2D([], [], color=INK_2, linewidth=2.2),
        plt.Line2D([], [], color=GRID, linewidth=1.4, linestyle="--"),
        plt.Line2D([], [], color=INK, linewidth=1.3),
    ]
    fig.legend(
        handles,
        [
            "FLDAS actual, next 30 days (state color)",
            "persistence (last 30 days)",
            "VCI-only model (mean of 3 seeds)",
        ],
        frameon=False,
        labelcolor=INK,
        fontsize=8,
        ncol=3,
        loc="upper center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG / "test_daily_series.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)
    daily = load_daily()
    start = first_anchor(daily)
    anchors = np.arange(start, len(daily["days"]))
    coarse = samples_for(daily, anchors[::TRAIN_STRIDE])
    fitset = select(coarse, np.isin(coarse["split"], ["train", "val"]))
    every_day = samples_for(daily, anchors)
    test = select(every_day, every_day["split"] == "test")
    del every_day
    print(
        "first anchor",
        daily["days"][start],
        "| samples:",
        {s: int((fitset["split"] == s).sum()) for s in ("train", "val")},
        "test",
        len(test["y"]),
    )

    D.INPUTS = ["vci"]
    n_eff = D.effective_cells(select(fitset, fitset["split"] == "train"))
    shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}
    held, n_blocks, n_held = holdout_blocks(fitset)
    held_test = holdout_blocks(test)[0]
    month_end = pd.DatetimeIndex(test["target_month"]).is_month_end

    rows, gaps, runs, preds = [], {}, {}, []
    for seed in SEEDS:
        run = D.fit(level(fitset), shares, f"vci_daily_seed{seed}", seed, WEIGHT_DECAY)
        pred = D.forecast(run["model"], level(test), np.ones(len(test["y"]), bool))
        preds.append(pred)
        rows += evaluate(pred, test, "test_daily", seed)
        rows += evaluate(pred[month_end], select(test, month_end), "test_month_end", seed)
        gaps[f"seed{seed}"] = D.bootstrap_gap(pred, by_month(test), np.ones(len(pred), bool))
        spatial = D.fit(
            level(fitset), shares, f"vci_daily_seed{seed}_spatial", seed, WEIGHT_DECAY, held
        )
        unseen = select(test, held_test)
        upred = D.forecast(spatial["model"], level(unseen), np.ones(len(unseen["y"]), bool))
        rows += evaluate(upred, unseen, "spatial_unseen_daily", seed)
        torch.save(run["model"].state_dict(), OUT / f"scratch_bilstm_vci_daily_seed{seed}.pt")
        runs[seed] = {k: run[k] for k in ("best_epoch", "epochs_run", "best_val", "seconds")}
        runs[seed]["spatial"] = {k: spatial[k] for k in ("best_epoch", "epochs_run", "seconds")}
        if seed == SEEDS[0]:
            runs["equivalence_pass"] = equivalence(run["model"])[1]
        print(seed, runs[seed], flush=True)

    ensemble = np.mean(preds, axis=0)
    rows += [
        r | {"seed": "mean"}
        for r in evaluate(ensemble, test, "test_daily", SEEDS[0])
        if r["model"] == "vci_daily"
    ]
    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "metrics_by_seed.csv", index=False)
    series = (
        pd.DataFrame(
            {
                "state": np.array(STATE_NAMES)[test["state"]],
                "anchor": test["target_month"],
                "actual": test["y"],
                "persistence": test["persistence"],
                "prediction": ensemble,
            }
        )
        .groupby(["state", "anchor"])
        .mean()
        .reset_index()
    )
    series.round(4).to_csv(METRICS / "test_daily_state_mean.csv", index=False)
    summary = {
        "effective_cells": n_eff,
        "state_shares": shares,
        "runs": runs,
        "spatial": {"blocks": n_blocks, "held_out_blocks": n_held},
        "mse_gap_vs_persistence": gaps,
        "data": json.loads(DAILY_SUMMARY.read_text()),
        "train_stride_days": TRAIN_STRIDE,
        "first_anchor": str(daily["days"][start]),
    }
    (METRICS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    figure_series(series)
    view = table[table.seed.astype(str).isin(["0", "mean"]) | (table.model == "vci_daily")]
    print(view.groupby(["run", "model", "group"]).mse.mean().unstack().round(4))
    assert runs["equivalence_pass"], "equivalence check FAILED"


if __name__ == "__main__":
    main()
