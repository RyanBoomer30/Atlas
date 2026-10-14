"""Split v2 + SPEI history, trained on the change from persistence (see RESULTS_v2.md).

Changes from ``train_v2.py --spei-history``:

- Target is dSPEI-12 = SPEI-12(t+1) - SPEI-12(t); the forecast is SPEI-12(t) + output.
  A model that outputs 0 is persistence. MSE on the change equals MSE on the level, so
  ``BiLSTM.training_step`` is unchanged.
- Weight decay (AdamW, chosen on val with seed 42 from WEIGHT_DECAYS) and early stopping
  on val_loss; the best-val checkpoint is still the one kept.
- Each state's share of training draws is proportional to its effective number of
  independent cells (from the train-year dSPEI correlation between cells), not 1/3.
- Seeds 42, 43, 44, each with its own spatial-holdout model; test metrics are reported
  per seed and as mean/min/max, with a month-block bootstrap interval on the MSE
  difference from persistence.

    python train_v2_delta.py
"""

import json
import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bilstm import BiLSTM
from data_v2 import FEATURES, SPEI_HISTORY_FEATURE, STATE_NAMES, load_samples
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from train import INK, INK_2, SURFACE, style
from train_v2 import BATCH, EPOCHS, HIDDEN, LABELS, LR, STATE_COLORS, equivalence, scores
from train_v2 import spatial_holdout as holdout_blocks

OUT = Path(__file__).parent
FIG = OUT / "figures_v2_delta"
METRICS = OUT / "metrics_v2_delta"
INPUTS = [*FEATURES, SPEI_HISTORY_FEATURE]
SEEDS = (42, 43, 44)
WEIGHT_DECAYS = (0.0, 1e-3, 1e-2)
PATIENCE = 4
BOOTSTRAP = 2000


class RegularizedBiLSTM(BiLSTM):
    """The scratch BiLSTM, unchanged, trained with AdamW weight decay."""

    def __init__(self, input_size, hidden_size, lr=1e-3, weight_decay=0.0):
        super().__init__(input_size, hidden_size, lr)
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.weight_decay
        )


def effective_cells(samples):
    """Per state: effective number of independent cells, from train-year dSPEI.

    n_eff = (sum of eigenvalues)^2 / sum of squared eigenvalues of the cell-by-cell
    correlation matrix: n for independent cells, 1 if every cell moves together.
    """
    train = samples["split"] == "train"
    frame = pd.DataFrame(
        {
            "state": samples["state"][train],
            "cell": samples["lon"][train].astype(str) + "," + samples["lat"][train].astype(str),
            "month": samples["target_month"][train],
            "dy": samples["y"][train] - samples["persistence"][train],
        }
    )
    out = {}
    for i, state in enumerate(STATE_NAMES):
        wide = frame[frame.state == i].pivot(index="month", columns="cell", values="dy")
        eig = np.linalg.eigvalsh(np.corrcoef(wide.to_numpy().T))
        out[state] = float(eig.sum() ** 2 / (eig**2).sum())
    return out


def sampler(states, shares, seed):
    """Draw each state with probability ``shares[state]``, uniformly within a state."""
    counts = np.bincount(states, minlength=len(STATE_NAMES))
    share = np.array([shares[s] for s in STATE_NAMES])
    weights = torch.from_numpy(share[states] / counts[states])
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(states), replacement=True, generator=generator)


def datasets(samples, exclude=None):
    """Train/val TensorDatasets with the dSPEI target, and the train mask used."""
    delta = samples["y"] - samples["persistence"]
    out, masks = {}, {}
    for name in ("train", "val"):
        keep = samples["split"] == name
        if exclude is not None:
            keep &= ~exclude
        masks[name] = keep
        out[name] = TensorDataset(
            torch.from_numpy(samples["x"][keep]), torch.from_numpy(delta[keep]).unsqueeze(1)
        )
    return out, masks


def fit(samples, shares, name, seed, weight_decay, exclude=None, hidden=HIDDEN):
    L.seed_everything(seed, workers=True)
    data, masks = datasets(samples, exclude)
    loaders = {
        "train": DataLoader(
            data["train"],
            batch_size=BATCH,
            sampler=sampler(samples["state"][masks["train"]], shares, seed),
        ),
        "val": DataLoader(data["val"], batch_size=4096),
    }
    model = RegularizedBiLSTM(len(INPUTS), hidden, LR, weight_decay)
    logger = CSVLogger(OUT / "logs", name=name)
    best = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best")
    stop = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=[best, stop],
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=50,
        accelerator="cpu",
    )
    start = time.time()
    trainer.fit(model, loaders["train"], loaders["val"])
    seconds = time.time() - start
    checkpoint = torch.load(best.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {
        "model": model,
        "log_dir": Path(logger.log_dir),
        "best_epoch": checkpoint["epoch"] + 1,
        "epochs_run": trainer.current_epoch,
        "best_val": float(best.best_model_score),
        "seconds": round(seconds, 1),
    }


def forecast(model, samples, mask):
    """Level forecast: persistence plus the predicted change."""
    x = samples["x"][mask]
    with torch.no_grad():
        delta = torch.cat([model(torch.from_numpy(b)) for b in np.array_split(x, 64)])
    return samples["persistence"][mask] + delta.numpy()[:, 0]


def evaluate(pred, samples, mask, seed, run):
    y, persist, states = samples["y"][mask], samples["persistence"][mask], samples["state"][mask]
    rows = []
    for group in ["overall", *STATE_NAMES]:
        sel = np.ones(len(y), bool) if group == "overall" else states == STATE_NAMES.index(group)
        for model_name, guess in (("bilstm_delta", pred[sel]), ("persistence", persist[sel])):
            rows.append(
                {"run": run, "seed": seed, "group": group, "model": model_name, "n": int(sel.sum())}
                | scores(guess, y[sel])
            )
    return rows


def bootstrap_gap(pred, samples, mask, seed=0):
    """95% interval of MSE(model) - MSE(persistence), resampling whole target months.

    All cells of a month move together, so months, not samples, are the resampling unit.
    """
    y, persist = samples["y"][mask], samples["persistence"][mask]
    months, index = np.unique(samples["target_month"][mask], return_inverse=True)
    states = samples["state"][mask]
    out = {}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(months), size=(BOOTSTRAP, len(months)))
    for group in ["overall", *STATE_NAMES]:
        sel = np.ones(len(y), bool) if group == "overall" else states == STATE_NAMES.index(group)
        gap = (pred[sel] - y[sel]) ** 2 - (persist[sel] - y[sel]) ** 2
        per_month = np.bincount(index[sel], weights=gap, minlength=len(months))
        n_month = np.bincount(index[sel], minlength=len(months))
        sums = per_month[draws].sum(axis=1) / n_month[draws].sum(axis=1)
        out[group] = {
            "mse_gap": float(gap.mean()),
            "ci95": [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))],
        }
    return out


def figure_skill(table, previous):
    """Test MSE relative to persistence per state: level target vs change target (per seed)."""
    groups = ["overall", *STATE_NAMES]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    width = 0.36

    def label(x, v, color=INK_2):
        ax.text(
            x,
            v,
            f"{v:+.0%}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            color=color,
            fontsize=8,
        )

    level = [previous["v2_spei"][g] for g in groups]
    xs = np.arange(len(groups)) - width / 2
    ax.bar(xs, level, width - 0.04, color="#b5b3ad", edgecolor=SURFACE, linewidth=2)
    for x, v in zip(xs, level, strict=True):
        label(x, v)
    test = table[(table.run == "test") & (table.model == "bilstm_delta")]
    base = table[(table.run == "test") & (table.model == "persistence") & (table.seed == SEEDS[0])]
    base = base.set_index("group").mse
    xs = np.arange(len(groups)) + width / 2
    for x, group in zip(xs, groups, strict=True):
        rel = test[test.group == group].mse.to_numpy() / base[group] - 1
        ax.bar(
            x,
            rel.mean(),
            width - 0.04,
            color=STATE_COLORS.get(group, INK),
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.plot([x] * len(rel), rel, "o", color=INK, markersize=3.5, zorder=3)
        # The label is the seed mean, placed clear of the lowest seed's dot.
        ax.text(
            x,
            min(rel.min(), rel.mean()) - 0.012,
            f"{rel.mean():+.1%}",
            ha="center",
            va="top",
            color=INK,
            fontsize=8,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#b5b3ad"),
        plt.Rectangle((0, 0), 1, 1, color=INK),
        plt.Line2D([], [], color=INK, marker="o", linestyle="", markersize=3.5),
    ]
    ax.legend(
        handles,
        [
            "level target (train_v2 --spei-history)",
            "change target + weight decay (mean, state color)",
            "each seed",
        ],
        frameon=False,
        labelcolor=INK,
        loc="lower left",
        fontsize=8,
    )
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.text(-0.45, 0.005, "persistence", color=INK_2, va="bottom", ha="left", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([LABELS.get(g, "All states") for g in groups])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    style(
        ax,
        "Test MSE relative to persistence (below 0 = beats persistence)",
        "",
        "MSE vs persistence",
    )
    fig.tight_layout()
    fig.savefig(FIG / "skill_vs_persistence.png", dpi=160)
    plt.close(fig)


def figure_loss(runs):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    colors = {42: "#2a78d6", 43: "#eb6834", 44: "#1baf7a"}
    for seed, run in runs.items():
        metrics = pd.read_csv(run["log_dir"] / "metrics.csv")
        val = metrics.dropna(subset=["val_loss"])
        train = metrics.dropna(subset=["train_loss"])
        ax.plot(
            val.epoch + 1, val.val_loss, color=colors[seed], linewidth=2, label=f"val, seed {seed}"
        )
        ax.plot(train.epoch + 1, train.train_loss, color=colors[seed], linewidth=1, linestyle="--")
        ax.scatter(
            [run["best_epoch"]],
            [run["best_val"]],
            s=50,
            color=colors[seed],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
    ax.plot([], [], color=INK_2, linestyle="--", linewidth=1, label="train (dashed)")
    best = sorted({run["best_epoch"] for run in runs.values()})
    ax.text(
        0.97,
        0.6,
        f"best val epoch: {', '.join(map(str, best))} (dots)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
    )
    ax.legend(frameon=False, labelcolor=INK, fontsize=8)
    style(
        ax,
        "dSPEI target + weight decay: loss per epoch (early stopped)",
        "Epoch",
        "MSE (SPEI-12 units²)",
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(FIG / "loss_curves.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    METRICS.mkdir(exist_ok=True)
    samples = load_samples(spei_history=True)
    assert samples["x"].shape[2] == len(INPUTS)
    n_eff = effective_cells(samples)
    shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}
    print("effective cells:", n_eff, "shares:", shares)

    # 1. Weight decay chosen on val (seed 42 only); test is not looked at here.
    selection = []
    for wd in WEIGHT_DECAYS:
        run = fit(samples, shares, f"delta_wd{wd:g}_seed42", 42, wd)
        selection.append(
            {"weight_decay": wd}
            | {k: run[k] for k in ("best_val", "best_epoch", "epochs_run", "seconds")}
        )
        print(selection[-1])
    weight_decay = min(selection, key=lambda r: r["best_val"])["weight_decay"]
    pd.DataFrame(selection).to_csv(METRICS / "weight_decay_selection.csv", index=False)

    # 2. Seeds, test and spatial holdout.
    test = samples["split"] == "test"
    held, n_blocks, n_held = holdout_blocks(samples)
    rows, runs, gaps, summary_runs = [], {}, {}, {}
    for seed in SEEDS:
        run = fit(samples, shares, f"delta_seed{seed}", seed, weight_decay)
        runs[seed] = run
        pred = forecast(run["model"], samples, test)
        rows += evaluate(pred, samples, test, seed, "test")
        gaps[f"test_seed{seed}"] = bootstrap_gap(pred, samples, test)
        torch.save(run["model"].state_dict(), OUT / f"scratch_bilstm_v2_delta_seed{seed}.pt")

        spatial = fit(samples, shares, f"delta_seed{seed}_spatial", seed, weight_decay, held)
        unseen = test & held
        rows += evaluate(
            forecast(spatial["model"], samples, unseen), samples, unseen, seed, "spatial_unseen"
        )
        rows += evaluate(
            forecast(run["model"], samples, unseen), samples, unseen, seed, "spatial_seen"
        )
        gaps[f"spatial_unseen_seed{seed}"] = bootstrap_gap(
            forecast(spatial["model"], samples, unseen), samples, unseen
        )
        summary_runs[seed] = {
            k: run[k] for k in ("best_epoch", "epochs_run", "best_val", "seconds")
        } | {"spatial": {k: spatial[k] for k in ("best_epoch", "epochs_run", "seconds")}}
        print(seed, summary_runs[seed])

    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "metrics_by_seed.csv", index=False)
    agg = (
        table.groupby(["run", "group", "model"])
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
    print(agg.round(4).to_string(index=False))

    results, ok = equivalence(runs[42]["model"])
    (METRICS / "summary.json").write_text(
        json.dumps(
            {
                "target": "dSPEI-12 = SPEI-12(t+1) - SPEI-12(t); forecast = SPEI-12(t) + output",
                "features": INPUTS,
                "hidden_size": HIDDEN,
                "batch_size": BATCH,
                "lr": LR,
                "max_epochs": EPOCHS,
                "early_stopping_patience": PATIENCE,
                "weight_decay": weight_decay,
                "weight_decay_selection": selection,
                "effective_cells": n_eff,
                "state_shares": shares,
                "seeds": summary_runs,
                "spatial": {"blocks": n_blocks, "held_out_blocks": n_held},
                "mse_gap_vs_persistence": gaps,
                "equivalence_seed42": {
                    "pass": ok,
                    "max_abs_diff": {
                        dtype: max(diff for _, diff in out.values())
                        for dtype, out in results.items()
                    },
                },
            },
            indent=2,
        )
    )

    previous = {}
    for tag in ("v2_spei",):
        old = pd.read_csv(OUT / f"metrics_{tag}/test_metrics.csv").set_index(["group", "model"])
        previous[tag] = {
            g: old.loc[(g, "bilstm"), "mse"] / old.loc[(g, "persistence"), "mse"] - 1
            for g in ["overall", *STATE_NAMES]
        }
    figure_skill(table, previous)
    figure_loss(runs)
    print("equivalence:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit("equivalence check FAILED")


if __name__ == "__main__":
    main()
