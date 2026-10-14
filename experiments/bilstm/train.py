"""Train the scratch BiLSTM on the shared split (seed 42, hidden 64, 30 epochs).

Re-runs the nn.LSTM equivalence check on the trained weights and saves figures to
./figures.
"""

from pathlib import Path

import check_equivalence
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
from bilstm import BiLSTM
from data import FEATURES, load_splits
from lightning.pytorch.loggers import CSVLogger
from shared_split import SEED, SPLIT_VERSION
from torch.utils.data import DataLoader

HIDDEN, EPOCHS, BATCH = 64, 30, 16
OUT = Path(__file__).parent
FIG = OUT / "figures"

# Reference palette (dataviz skill), light mode.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    L.seed_everything(SEED, workers=True)
    splits = load_splits()
    loaders = {
        name: DataLoader(ds, batch_size=BATCH, shuffle=(name == "train"))
        for name, ds in splits.items()
    }
    print(f"shared split {SPLIT_VERSION}:", {name: len(ds) for name, ds in splits.items()})

    model = BiLSTM(input_size=len(FEATURES), hidden_size=HIDDEN)
    logger = CSVLogger(OUT / "logs", name="scratch_bilstm")
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        deterministic=True,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
        accelerator="cpu",
    )
    trainer.fit(model, loaders["train"], loaders["val"])
    torch.save(model.state_dict(), OUT / "scratch_bilstm.pt")

    model.eval()
    x_test, y_test = splits["test"].tensors
    with torch.no_grad():
        pred = model(x_test)
    test_mse = torch.mean((pred - y_test) ** 2).item()
    # Reference: always predict the train-set mean SPEI-12.
    train_mean = splits["train"].tensors[1].mean()
    baseline_mse = torch.mean((train_mean - y_test) ** 2).item()
    print(f"test MSE: {test_mse:.4f}   (predict-train-mean baseline: {baseline_mse:.4f})")

    print("\nEquivalence on trained weights:")
    results = check_equivalence.run(model=model, dtype=torch.float32)

    FIG.mkdir(exist_ok=True)
    metrics = pd.read_csv(Path(logger.log_dir) / "metrics.csv")
    train = metrics.dropna(subset=["train_loss"])
    val = metrics.dropna(subset=["val_loss"])

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.plot(train["epoch"] + 1, train["train_loss"], color=SERIES_1, linewidth=2)
    ax.plot(val["epoch"] + 1, val["val_loss"], color=SERIES_2, linewidth=2)
    ax.annotate(
        "train",
        (train["epoch"].iloc[-1] + 1, train["train_loss"].iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK,
        va="center",
    )
    ax.annotate(
        "val",
        (val["epoch"].iloc[-1] + 1, val["val_loss"].iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK,
        va="center",
    )
    ax.legend(["train", "val"], frameon=False, labelcolor=INK)
    style(ax, "Scratch BiLSTM: MSE loss per epoch", "Epoch", "MSE (SPEI-12 units²)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG / "loss_curves.png", dpi=160)

    # 2. Predicted vs actual on test
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    lo = min(y_test.min(), pred.min()).item() - 0.2
    hi = max(y_test.max(), pred.max()).item() + 0.2
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1, linestyle="--")
    ax.scatter(y_test, pred, s=40, color=SERIES_1, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(
        0.04,
        0.96,
        f"test MSE = {test_mse:.3f}\nmean baseline = {baseline_mse:.3f}\nn = {len(y_test)}",
        transform=ax.transAxes,
        va="top",
        color=INK,
    )
    style(
        ax,
        f"Test set ({SPLIT_VERSION}): predicted vs actual",
        "Actual next-month SPEI-12",
        "Predicted SPEI-12",
    )
    fig.tight_layout()
    fig.savefig(FIG / "pred_vs_actual.png", dpi=160)

    # 3. Equivalence: max |diff| vs nn.LSTM per case, against the allclose tolerance
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    labels = [f"{name}\n{'split bias' if split else 'one bias'}" for name, split in results]
    diffs = [max(d, 1e-12) for _, d in results.values()]
    ax.bar(labels, diffs, color=SERIES_1, width=0.6)
    ax.axhline(1e-6, color=INK_2, linestyle="--", linewidth=1)
    ax.text(len(labels) - 0.5, 1e-6, "atol 1e-6", color=INK_2, va="bottom", ha="right")
    ax.set_yscale("log")
    style(ax, "Trained weights vs nn.LSTM: max |output diff| (float32)", "", "max |diff|")
    fig.tight_layout()
    fig.savefig(FIG / "equivalence.png", dpi=160)
    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
