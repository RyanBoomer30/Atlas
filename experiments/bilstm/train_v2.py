"""Train the scratch BiLSTM on shared split v2 and evaluate it (see SPLIT_v2.md).

Seed 42, hidden 64, Adam 1e-3, MSE, at most 30 epochs, batch 256 (v1 used 16), states
balanced by sampling. Keeps the checkpoint with the best val_loss, reports test
metrics overall and per state against the train-mean and persistence baselines,
runs the 2x2 deg spatial block holdout, re-runs the nn.LSTM equivalence check on the
trained weights, and writes figures to ./figures_v2 and metrics to ./metrics_v2.

    python train_v2.py
"""

import copy
import json
import time
from pathlib import Path

import check_equivalence
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bilstm import BiLSTM
from data_v2 import CELLS_CSV, FEATURES, MASKS_JSON, STATE_NAMES, load_samples, load_splits
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from shared_split_v2 import BLOCK_HOLDOUT_FRACTION, SEED, SPLIT_VERSION, block_of
from torch.utils.data import DataLoader, WeightedRandomSampler
from train import GRID, INK, INK_2, SURFACE, style

HIDDEN, EPOCHS, BATCH, LR = 64, 30, 256, 1e-3
OUT = Path(__file__).parent
FIG = OUT / "figures_v2"
METRICS = OUT / "metrics_v2"
STATE_COLORS = {"kansas": "#2a78d6", "texas": "#eb6834", "california": "#1baf7a"}
LABELS = {"kansas": "Kansas", "texas": "Texas", "california": "California"}
# Baselines are neutral so the state hue stays with the model.
BASELINE_COLORS = {"train mean": "#b5b3ad", "persistence": "#7a7873"}


def balanced_sampler(states, generator):
    """Each state gets an equal share of every training epoch, drawn with replacement."""
    counts = np.bincount(states, minlength=len(STATE_NAMES))
    weights = torch.from_numpy(1.0 / counts[states])
    return WeightedRandomSampler(weights, len(states), replacement=True, generator=generator)


def fit(samples, name, exclude=None):
    """Train one model; return it (best val_loss weights), its log dir and the best epoch."""
    L.seed_everything(SEED, workers=True)
    splits = load_splits(samples, exclude)
    train_mask = samples["split"] == "train"
    if exclude is not None:
        train_mask &= ~exclude
    generator = torch.Generator().manual_seed(SEED)
    loaders = {
        "train": DataLoader(
            splits["train"],
            batch_size=BATCH,
            sampler=balanced_sampler(samples["state"][train_mask], generator),
        ),
        "val": DataLoader(splits["val"], batch_size=4096),
    }
    print(f"[{name}] samples:", {k: len(v) for k, v in splits.items()})
    model = BiLSTM(input_size=len(FEATURES), hidden_size=HIDDEN, lr=LR)
    logger = CSVLogger(OUT / "logs", name=name)
    best = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best")
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=[best],
        deterministic=True,
        enable_progress_bar=False,
        log_every_n_steps=50,
        accelerator="cpu",
    )
    start = time.time()
    trainer.fit(model, loaders["train"], loaders["val"])
    seconds = time.time() - start
    checkpoint = torch.load(best.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, Path(logger.log_dir), checkpoint["epoch"] + 1, seconds


def predict(model, x):
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(b)) for b in np.array_split(x, 64)]).numpy()[:, 0]


def scores(pred, y):
    err = pred - y
    r = np.corrcoef(pred, y)[0, 1] if np.std(pred) > 0 else float("nan")
    return {"mse": float(np.mean(err**2)), "mae": float(np.mean(np.abs(err))), "r": float(r)}


def evaluate(model, samples, mask, train_mean):
    """Model and both baselines, overall and per state, on the samples in ``mask``."""
    pred = predict(model, samples["x"][mask])
    y, persist, states = samples["y"][mask], samples["persistence"][mask], samples["state"][mask]
    rows = []
    for group in ["overall", *STATE_NAMES]:
        sel = np.ones(len(y), bool) if group == "overall" else states == STATE_NAMES.index(group)
        if not sel.any():
            continue
        for model_name, guess in (
            ("bilstm", pred[sel]),
            ("train mean", np.full(sel.sum(), train_mean)),
            ("persistence", persist[sel]),
        ):
            rows.append(
                {"group": group, "model": model_name, "n": int(sel.sum())} | scores(guess, y[sel])
            )
    return pd.DataFrame(rows), pred


def spatial_holdout(samples):
    """Boolean mask of samples in a random BLOCK_HOLDOUT_FRACTION of 2x2 deg blocks."""
    blocks = np.array(
        [block_of(lon, lat) for lon, lat in zip(samples["lon"], samples["lat"], strict=True)]
    )
    unique = np.unique(blocks, axis=0)
    rng = np.random.default_rng(SEED)
    chosen = unique[rng.permutation(len(unique))[: round(BLOCK_HOLDOUT_FRACTION * len(unique))]]
    held = (blocks[:, None, :] == chosen[None, :, :]).all(axis=2).any(axis=1)
    return held, len(unique), len(chosen)


def equivalence(model):
    """check_equivalence on the trained weights in float32 and float64."""
    results = {}
    for dtype in (torch.float32, torch.float64):
        out = check_equivalence.run(model=copy.deepcopy(model), dtype=dtype)
        results[str(dtype).split(".")[1]] = out
    ok = all(ok for out in results.values() for ok, _ in out.values())
    print("equivalence:", "PASS" if ok else "FAIL")
    return results, ok


def direct_label(ax, x, y, text, color=INK):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", color=color, va="center")


def figure_loss(log_dir, best_epoch):
    metrics = pd.read_csv(log_dir / "metrics.csv")
    train = metrics.dropna(subset=["train_loss"])
    val = metrics.dropna(subset=["val_loss"])
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.plot(train["epoch"] + 1, train["train_loss"], color="#2a78d6", linewidth=2)
    ax.plot(val["epoch"] + 1, val["val_loss"], color="#eb6834", linewidth=2)
    direct_label(ax, train["epoch"].iloc[-1] + 1, train["train_loss"].iloc[-1], "train")
    direct_label(ax, val["epoch"].iloc[-1] + 1, val["val_loss"].iloc[-1], "val")
    best_val = val.loc[val["epoch"] + 1 == best_epoch, "val_loss"].iloc[0]
    ax.axvline(best_epoch, color=INK_2, linestyle="--", linewidth=1)
    ax.scatter(
        [best_epoch], [best_val], s=64, color="#eb6834", edgecolor=SURFACE, linewidth=2, zorder=3
    )
    ax.annotate(
        f"best val epoch {best_epoch}\nval MSE {best_val:.3f}",
        (best_epoch, best_val),
        xytext=(10, -34),
        textcoords="offset points",
        color=INK,
    )
    ax.legend(["train (state-balanced)", "val"], frameon=False, labelcolor=INK)
    style(
        ax, f"Scratch BiLSTM ({SPLIT_VERSION}): MSE loss per epoch", "Epoch", "MSE (SPEI-12 units²)"
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG / "loss_curves.png", dpi=160)
    plt.close(fig)


def figure_pred_vs_actual(samples, test, pred, table):
    fig, axes = plt.subplots(
        1,
        len(STATE_NAMES),
        figsize=(13, 4.6),
        facecolor=SURFACE,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    y, states = samples["y"][test], samples["state"][test]
    lo, hi = min(y.min(), pred.min()) - 0.2, max(y.max(), pred.max()) + 0.2
    for i, (ax, state) in enumerate(zip(axes, STATE_NAMES, strict=True)):
        sel = states == i
        ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--")
        ax.scatter(
            y[sel],
            pred[sel],
            s=8,
            alpha=0.25,
            color=STATE_COLORS[state],
            linewidth=0,
            label=LABELS[state],
        )
        row = table[(table.group == state) & (table.model == "bilstm")].iloc[0]
        ax.text(
            0.04,
            0.96,
            f"{LABELS[state]}\nMSE {row.mse:.3f}  r {row.r:.2f}\nn = {row.n}",
            transform=ax.transAxes,
            va="top",
            color=INK,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        style(ax, "", "Actual next-month SPEI-12", "Predicted SPEI-12" if i == 0 else "")
        ax.legend(loc="lower right", frameon=False, labelcolor=INK, markerscale=3)
    fig.suptitle(
        f"Test set ({SPLIT_VERSION}, 2021-2024): predicted vs actual",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG / "pred_vs_actual.png", dpi=110)  # dense scatter; keeps the PNG < 500 KB
    plt.close(fig)


def figure_metric_bars(table, name="metrics_by_state.png", title="Test"):
    groups = [g for g in ["overall", *STATE_NAMES] if g in set(table.group)]
    models = ["bilstm", "train mean", "persistence"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    width = 0.26
    for ax, metric, label in (
        (axes[0], "mse", "MSE (lower is better)"),
        (axes[1], "r", "Pearson r (higher is better)"),
    ):
        for j, model_name in enumerate(models):
            for i, group in enumerate(groups):
                value = table[(table.group == group) & (table.model == model_name)][metric]
                value = float(value.iloc[0]) if len(value) else float("nan")
                if model_name == "bilstm":
                    color = STATE_COLORS.get(group, INK)
                else:
                    color = BASELINE_COLORS[model_name]
                x = i + (j - 1) * width
                ax.bar(
                    x,
                    0 if np.isnan(value) else value,
                    width - 0.03,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=2,
                )
                text = "n/a" if np.isnan(value) else f"{value:.2f}"
                ax.text(
                    x,
                    (0 if np.isnan(value) else max(value, 0)),
                    text,
                    ha="center",
                    va="bottom",
                    color=INK_2,
                    fontsize=7,
                )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([LABELS.get(g, "All states") for g in groups])
        ax.axhline(0, color=GRID, linewidth=1)
        style(ax, label, "", "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=INK)] + [
        plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLORS[m]) for m in models[1:]
    ]
    fig.legend(
        handles,
        ["BiLSTM (state color)", "train mean", "persistence"],
        frameon=False,
        labelcolor=INK,
        loc="upper right",
        ncol=3,
    )
    fig.suptitle(
        f"{title}: BiLSTM vs baselines by state", x=0.01, ha="left", color=INK, fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def figure_map(held_blocks):
    cells = pd.read_csv(CELLS_CSV)
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    dropped = cells[~cells.kept]
    ax.scatter(
        dropped.longitude,
        dropped.latitude,
        s=14,
        color=GRID,
        marker="s",
        linewidth=0,
        label="masked out",
    )
    for state in STATE_NAMES:
        kept = cells[cells.kept & (cells.state == state)]
        ax.scatter(
            kept.longitude,
            kept.latitude,
            s=14,
            marker="s",
            linewidth=0,
            color=STATE_COLORS[state],
            label=f"{LABELS[state]} ({len(kept)})",
        )
        # Texas sits just south of Kansas, so its label goes under its cells.
        y = kept.latitude.min() - 1.0 if state == "texas" else kept.latitude.max() + 0.6
        ax.text(kept.longitude.mean(), y, LABELS[state], color=INK, ha="center")
    for bx, by in held_blocks:
        ax.add_patch(
            plt.Rectangle(
                (bx * 2, by * 2), 2, 2, fill=False, edgecolor=INK_2, linewidth=0.8, linestyle=":"
            )
        )
    ax.plot([], [], color=INK_2, linestyle=":", label="spatial-holdout block")
    ax.set_aspect("equal")
    ax.legend(frameon=False, labelcolor=INK, loc="lower left", fontsize=8)
    style(ax, f"Kept 0.5° SPEI cells ({SPLIT_VERSION})", "Longitude", "Latitude")
    fig.tight_layout()
    fig.savefig(FIG / "cells_map.png", dpi=160)
    plt.close(fig)


def figure_equivalence(results):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    cases = list(results["float32"])
    labels = [f"{name}\n{'split bias' if split else 'one bias'}" for name, split in cases]
    width = 0.38
    for j, (dtype, color, atol) in enumerate(
        (("float32", "#2a78d6", 1e-6), ("float64", "#eb6834", 1e-12))
    ):
        diffs = [max(results[dtype][c][1], 1e-18) for c in cases]
        xs = np.arange(len(cases)) + (j - 0.5) * width
        ax.bar(
            xs,
            diffs,
            width - 0.03,
            color=color,
            label=f"{dtype} (atol {atol:g})",
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.axhline(atol, color=color, linestyle="--", linewidth=1)
        ax.text(
            len(cases) - 0.5, atol, f"{dtype} atol {atol:g}", color=INK_2, va="bottom", ha="right"
        )
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.legend(frameon=False, labelcolor=INK, loc="center left")
    style(ax, "Trained v2 weights vs nn.LSTM: max |output diff|", "", "max |diff|")
    fig.tight_layout()
    fig.savefig(FIG / "equivalence.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)
    samples = load_samples()
    split = samples["split"]
    counts = (
        pd.DataFrame({"state": np.array(STATE_NAMES)[samples["state"]], "split": split})
        .query("split != 'gap'")
        .value_counts()
        .unstack(fill_value=0)[["train", "val", "test"]]
    )
    counts["cells"] = pd.read_csv(CELLS_CSV).query("kept").groupby("state").size()
    counts.to_csv(METRICS / "counts.csv")
    print(counts)

    model, log_dir, best_epoch, seconds = fit(samples, "scratch_bilstm_v2")
    torch.save(model.state_dict(), OUT / "scratch_bilstm_v2.pt")
    train_mean = float(samples["y"][split == "train"].mean())
    test = split == "test"
    table, pred = evaluate(model, samples, test, train_mean)
    table.to_csv(METRICS / "test_metrics.csv", index=False)
    print(table.to_string(index=False))

    held, n_blocks, n_held = spatial_holdout(samples)
    spatial_model, _, spatial_epoch, spatial_seconds = fit(
        samples, "scratch_bilstm_v2_spatial", exclude=held
    )
    spatial_train_mean = float(samples["y"][(split == "train") & ~held].mean())
    spatial, _ = evaluate(spatial_model, samples, test & held, spatial_train_mean)
    spatial.to_csv(METRICS / "spatial_holdout_metrics.csv", index=False)
    # Same model as the main run, scored on the same held-out cells, for comparison.
    seen, _ = evaluate(model, samples, test & held, train_mean)
    seen.to_csv(METRICS / "spatial_cells_main_model_metrics.csv", index=False)
    print(spatial.to_string(index=False))

    results, ok = equivalence(model)
    blocks = np.array(
        [
            block_of(lon, lat)
            for lon, lat in zip(samples["lon"][held], samples["lat"][held], strict=True)
        ]
    )
    summary = {
        "split_version": SPLIT_VERSION,
        "features": FEATURES,
        "hidden_size": HIDDEN,
        "batch_size": BATCH,
        "lr": LR,
        "max_epochs": EPOCHS,
        "best_epoch": best_epoch,
        "train_seconds": round(seconds, 1),
        "train_mean": train_mean,
        "spatial": {
            "blocks": n_blocks,
            "held_out_blocks": n_held,
            "held_out_cells": len(
                np.unique(np.stack([samples["lon"][held], samples["lat"][held]]), axis=1).T
            ),
            "best_epoch": spatial_epoch,
            "train_seconds": round(spatial_seconds, 1),
        },
        "equivalence": {
            "pass": ok,
            **{
                dtype: {
                    f"{name}_{'split' if s else 'one'}_bias": diff
                    for (name, s), (_, diff) in out.items()
                }
                for dtype, out in results.items()
            },
        },
        "masks": json.loads(MASKS_JSON.read_text()),
    }
    (METRICS / "summary.json").write_text(json.dumps(summary, indent=2))

    figure_loss(log_dir, best_epoch)
    figure_pred_vs_actual(samples, test, pred, table)
    figure_metric_bars(table)
    figure_metric_bars(
        spatial,
        "spatial_holdout_metrics.png",
        "Spatial holdout (unseen 2x2 deg blocks, test years)",
    )
    figure_map({tuple(b) for b in np.unique(blocks, axis=0)})
    figure_equivalence(results)
    print(f"figures written to {FIG}")
    if not ok:
        raise SystemExit("equivalence check FAILED")


if __name__ == "__main__":
    main()
