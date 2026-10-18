"""VCI-only daily model trained on 2001-2023 and tested on 2024 (fixing the level offset).

The model in train_vci_daily.py learned from 2001-2014, so it centres its forecasts near
"normal" and misses that 2021-2024 was mostly dry. Here the training period runs up to
the test year, as in deployment. Inputs, labels, recipe and seeds are unchanged; only the
periods move:

1. Epoch count: train on label windows in 2001-01-01 .. 2022-12-31, validate on 2023, and
   take each seed's best epoch.
2. Final models: retrain on every window in 2001-01-01 .. 2023-12-31 for that many epochs
   (no validation data held back).
3. Test: forecasts issued every day whose 30-day window lies in 2024. The 2001-2014 models
   from train_vci_daily.py are scored on the same days for comparison.

    python train_vci_daily_recent.py
"""

import json

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import train_v2_delta as D
from data_vci_daily import LEAD_DAYS, first_anchor, load_daily, samples_for
from lightning.pytorch.loggers import CSVLogger
from predict_vci_daily import load_models as load_2014_models
from torch.utils.data import DataLoader, TensorDataset
from train import GRID, INK, INK_2, SURFACE, style
from train_v2 import LABELS, STATE_COLORS, STATE_NAMES, equivalence
from train_vci_daily import TRAIN_STRIDE, WEIGHT_DECAY, level, select

OUT = D.OUT
FIG = OUT / "figures_vci_daily_recent"
METRICS = OUT / "metrics_vci_daily_recent"
SEEDS = (42, 43, 44)
PERIODS = {
    "fit": ("2001-01-01", "2022-12-31"),
    "val": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2024-12-31"),
}
FULL_TRAIN = ("2001-01-01", "2023-12-31")


def window_period(samples, periods):
    """Label each sample by the period that contains its whole 30-day window, else 'none'."""
    anchor = pd.DatetimeIndex(samples["target_month"])
    start = (anchor + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (anchor + pd.Timedelta(days=LEAD_DAYS)).strftime("%Y-%m-%d")
    out = np.full(len(anchor), "none", dtype="U5")
    for name, (lo, hi) in periods.items():
        out[(start >= lo) & (end <= hi)] = name
    return out


def fit_fixed(samples, shares, name, seed, epochs):
    """Train for exactly ``epochs`` epochs on every sample (no validation, last weights kept)."""
    L.seed_everything(seed, workers=True)
    data = TensorDataset(
        torch.from_numpy(samples["x"]),
        torch.from_numpy(samples["y"] - samples["persistence"]).unsqueeze(1),
    )
    loader = DataLoader(data, batch_size=D.BATCH, sampler=D.sampler(samples["state"], shares, seed))
    model = D.RegularizedBiLSTM(len(D.INPUTS), D.HIDDEN, D.LR, WEIGHT_DECAY)
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=CSVLogger(OUT / "logs", name=name),
        enable_checkpointing=False,
        limit_val_batches=0,
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=50,
        accelerator="cpu",
    )
    trainer.fit(model, loader)
    model.eval()
    return model


def predict(models, x):
    with torch.no_grad():
        return np.mean(
            [
                torch.cat([m(torch.from_numpy(b)) for b in np.array_split(x, 32)]).numpy()[:, 0]
                for m in models
            ],
            axis=0,
        )


def decompose(pred, y):
    """R2 = r2 - scale penalty - offset penalty, plus MSE and skill vs climatology."""
    r = np.corrcoef(pred, y)[0, 1]
    sy, sp = y.std(), pred.std()
    mse = float(np.mean((pred - y) ** 2))
    return {
        "mse": mse,
        "R2": 1 - mse / y.var(),
        "r2": r * r,
        "scale_penalty": (r - sp / sy) ** 2,
        "offset_penalty": ((pred.mean() - y.mean()) / sy) ** 2,
        "skill_vs_climatology": 1 - mse / np.mean(y**2),
        "mean_pred": pred.mean(),
        "mean_actual": y.mean(),
        "sd_pred": sp,
        "sd_actual": sy,
    }


def figure(series):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.4), facecolor=SURFACE, sharex=True)
    for ax, state in zip(axes, STATE_NAMES, strict=True):
        m = series[series.state == state]
        when = pd.DatetimeIndex(m.anchor)
        ax.plot(when, m.actual, color=STATE_COLORS[state], linewidth=2.2)
        ax.plot(when, m.persistence, color=GRID, linewidth=1.4, linestyle="--")
        ax.plot(when, m.trained_2014, color="#9c9a94", linewidth=1.3, linestyle=":")
        ax.plot(when, m.trained_2023, color=INK, linewidth=1.5)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        style(ax, f"{LABELS[state]}: 2024 forecasts for the next 30 days", "", "SM anomaly (z)")
    handles = [
        plt.Line2D([], [], color=INK_2, linewidth=2.2),
        plt.Line2D([], [], color=GRID, linewidth=1.4, linestyle="--"),
        plt.Line2D([], [], color="#9c9a94", linewidth=1.3, linestyle=":"),
        plt.Line2D([], [], color=INK, linewidth=1.5),
    ]
    fig.legend(
        handles,
        [
            "FLDAS actual (state color)",
            "persistence",
            "VCI model trained 2001-2014",
            "VCI model trained 2001-2023",
        ],
        frameon=False,
        labelcolor=INK,
        fontsize=8,
        ncol=4,
        loc="upper center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG / "test_2024_series.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)
    D.INPUTS = ["vci"]
    daily = load_daily()
    anchors = np.arange(first_anchor(daily), len(daily["days"]))
    coarse = samples_for(daily, anchors[::TRAIN_STRIDE])
    coarse["split"] = window_period(coarse, PERIODS)
    phase1 = select(coarse, np.isin(coarse["split"], ["fit", "val"]))
    phase1["split"] = np.where(phase1["split"] == "fit", "train", "val")
    full = select(coarse, window_period(coarse, {"train": FULL_TRAIN}) == "train")
    every_day = samples_for(daily, anchors)
    test = select(every_day, window_period(every_day, PERIODS) == "test")
    del every_day
    print(
        "samples: phase1",
        {s: int((phase1["split"] == s).sum()) for s in ("train", "val")},
        "| full",
        len(full["y"]),
        "| test 2024",
        len(test["y"]),
        flush=True,
    )

    n_eff = D.effective_cells(full | {"split": np.full(len(full["y"]), "train", dtype="U5")})
    shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}

    runs, phase1_models, final_models = {}, [], []
    for seed in SEEDS:
        run = D.fit(level(phase1), shares, f"vci_recent_phase1_seed{seed}", seed, WEIGHT_DECAY)
        epochs = run["best_epoch"]
        phase1_models.append(run["model"])
        model = fit_fixed(level(full), shares, f"vci_recent_final_seed{seed}", seed, epochs)
        final_models.append(model)
        torch.save(model.state_dict(), OUT / f"scratch_bilstm_vci_daily_recent_seed{seed}.pt")
        runs[seed] = {
            "phase1_best_epoch": epochs,
            "phase1_best_val_2023": run["best_val"],
            "phase1_epochs_run": run["epochs_run"],
            "phase1_seconds": run["seconds"],
        }
        print(seed, runs[seed], flush=True)
    runs["equivalence_pass"] = equivalence(final_models[0])[1]

    y, persist, states = test["y"], test["persistence"], test["state"]
    preds = {
        "trained_2014": predict(load_2014_models(), test["x"]),
        "trained_2022": predict(phase1_models, test["x"]),
        "trained_2023": predict(final_models, test["x"]),
        "persistence": persist,
        "climatology": np.zeros_like(y),
    }
    per_seed = {
        f"trained_2023_seed{s}": predict([m], test["x"])
        for s, m in zip(SEEDS, final_models, strict=True)
    }
    rows = []
    for name, pred in {**preds, **per_seed}.items():
        for group in ["overall", *STATE_NAMES]:
            sel = (
                np.ones(len(y), bool) if group == "overall" else states == STATE_NAMES.index(group)
            )
            rows.append(
                {"model": name, "group": group, "n": int(sel.sum())}
                | decompose(pred[sel].astype(float), y[sel].astype(float))
            )
    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "test_2024.csv", index=False)
    series = (
        pd.DataFrame(
            {
                "state": np.array(STATE_NAMES)[states],
                "anchor": test["target_month"],
                "actual": y,
                **{k: preds[k] for k in ("persistence", "trained_2014", "trained_2023")},
            }
        )
        .groupby(["state", "anchor"])
        .mean()
        .reset_index()
    )
    series.round(4).to_csv(METRICS / "test_2024_state_mean.csv", index=False)
    (METRICS / "summary.json").write_text(
        json.dumps(
            {
                "periods": PERIODS,
                "full_train": FULL_TRAIN,
                "effective_cells": n_eff,
                "state_shares": shares,
                "runs": runs,
            },
            indent=2,
            default=str,
        )
    )
    figure(series)
    cols = [
        "mse",
        "R2",
        "r2",
        "scale_penalty",
        "offset_penalty",
        "skill_vs_climatology",
        "mean_pred",
        "mean_actual",
    ]
    for group in ["overall", *STATE_NAMES]:
        print(f"== {group}")
        print(
            table[(table.group == group) & ~table.model.str.contains("seed")]
            .set_index("model")[cols]
            .round(3)
            .to_string()
        )
    assert runs["equivalence_pass"], "equivalence check FAILED"


if __name__ == "__main__":
    main()
