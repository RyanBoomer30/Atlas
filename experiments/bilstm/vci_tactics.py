"""VCI-only tactics that leave the input unchanged, scored by rolling-origin testing.

The input is always the same daily VCI (12 x 30-day blocks). What varies:

- label: FLDAS soil-moisture anomaly for 0-10, 10-40, 40-100 cm or the 0-100 cm average
- recency weighting: training draws weighted by 2^((year - last train year) / HALF_LIFE)
- recalibration of the model output (post-processing):
    scale:  linear map y ~ a + b * prediction fitted on the training period
    offset: scale + shift by the mean error over the 12 months before the test year
            (uses soil-moisture labels already known when the forecast is issued)

Rolling origin: for each test year Y in 2021-2024, models train on every 30-day window in
2001-01-01 .. (Y-1)-12-31 and forecast every day of Y. Each model trains for one epoch
(every early-stopped VCI run picked epoch 1), seeds 42-44, ensemble = seed mean.

    python vci_tactics.py
"""

import json

import lightning as L
import numpy as np
import pandas as pd
import torch
import train_v2_delta as D
from data_v2 import FLDAS_FEATURES, MONTHS, STATE_NAMES, build_state
from data_vci import SM_LAYERS, SM_STATS_YEARS
from data_vci_daily import OUT_DIR, daily_labels, first_anchor, load_daily, samples_for
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from train_vci_daily import TRAIN_STRIDE, WEIGHT_DECAY
from train_vci_daily_recent import decompose, window_period

OUT = D.OUT
METRICS = OUT / "metrics_vci_tactics"
LABELS_CACHE = OUT_DIR / "daily_layer_labels.npz"
SEEDS = (42, 43, 44)
TEST_YEARS = (2021, 2022, 2023, 2024)
HALF_LIFE = 3.5  # years
LAYERS = {
    "sm_0_10": ["SoilMoi00_10cm_tavg"],
    "sm_10_40": ["SoilMoi10_40cm_tavg"],
    "sm_40_100": ["SoilMoi40_100cm_tavg"],
    "sm_0_100": list(SM_LAYERS),
}


def layer_labels(days):
    """Daily anomaly series per label key -> {key: (cells, days)} (cached)."""
    if LABELS_CACHE.exists():
        with np.load(LABELS_CACHE) as f:
            return {k: f[k] for k in f.files}
    years, cal = MONTHS.year.to_numpy(), MONTHS.month.to_numpy()
    stats = (years >= SM_STATS_YEARS[0]) & (years <= SM_STATS_YEARS[1])
    out = {k: [] for k in LAYERS}
    for state in STATE_NAMES:
        x, *_ = build_state(state)
        for key, layers in LAYERS.items():
            w = np.array([SM_LAYERS[n] for n in layers])
            v = sum(x[:, :, FLDAS_FEATURES.index(n)] * wi for n, wi in zip(layers, w, strict=True))
            v = v / w.sum()
            z = np.zeros_like(v)
            for m in range(1, 13):
                sel = cal == m
                ref = v[:, sel & stats]
                z[:, sel] = (v[:, sel] - ref.mean(1, keepdims=True)) / (
                    ref.std(1, keepdims=True) + 1e-9
                )
            out[key].append(daily_labels(z, days))
    out = {k: np.concatenate(v) for k, v in out.items()}
    np.savez(LABELS_CACHE, **out)
    return out


def fit_one_epoch(samples, weights, seed, name, num_samples=None):
    """One epoch of ``num_samples`` weighted draws (default: one per training sample)."""
    L.seed_everything(seed, workers=True)
    data = TensorDataset(
        torch.from_numpy(samples["x"]), torch.from_numpy(samples["y"]).unsqueeze(1)
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples or len(weights),
        replacement=True,
        generator=generator,
    )
    model = D.RegularizedBiLSTM(samples["x"].shape[2], D.HIDDEN, D.LR, WEIGHT_DECAY)
    trainer = L.Trainer(
        max_epochs=1,
        logger=CSVLogger(OUT / "logs", name=name),
        enable_checkpointing=False,
        limit_val_batches=0,
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=100,
        accelerator="cpu",
    )
    trainer.fit(model, DataLoader(data, batch_size=D.BATCH, sampler=sampler))
    model.eval()
    return model


def predict(models, x):
    with torch.no_grad():
        return np.mean(
            [
                torch.cat([m(torch.from_numpy(b)) for b in np.array_split(x, 64)]).numpy()[:, 0]
                for m in models
            ],
            axis=0,
        )


def state_weights(states, shares):
    counts = np.bincount(states, minlength=len(STATE_NAMES))
    share = np.array([shares[s] for s in STATE_NAMES])
    return share[states] / counts[states]


def select(samples, mask):
    return {k: v[mask] for k, v in samples.items()}


def main():
    METRICS.mkdir(exist_ok=True)
    D.INPUTS = ["vci"]
    daily = load_daily()
    days = pd.DatetimeIndex(daily["days"])
    labels = layer_labels(days)
    anchors = np.arange(first_anchor(daily), len(days))
    rows, fits = [], []
    for key in LAYERS:
        series = daily | {"sm": labels[key]}
        coarse = samples_for(series, anchors[::TRAIN_STRIDE])
        fine = samples_for(series, anchors)
        coarse_year = pd.DatetimeIndex(coarse["target_month"]).year.to_numpy()
        base_train = window_period(coarse, {"t": ("2001-01-01", "2020-12-31")}) == "t"
        n_eff = D.effective_cells(
            select(coarse, base_train) | {"split": np.full(base_train.sum(), "train", dtype="U5")}
        )
        shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}
        for year in TEST_YEARS:
            train = window_period(coarse, {"t": ("2001-01-01", f"{year - 1}-12-31")}) == "t"
            test = window_period(fine, {"t": (f"{year}-01-01", f"{year}-12-31")}) == "t"
            tr, te = select(coarse, train), select(fine, test)
            base_w = state_weights(tr["state"], shares)
            recency = 2.0 ** ((coarse_year[train] - (year - 1)) / HALF_LIFE)
            variants = {"uniform": base_w, "recency": base_w * recency}
            for weighting, w in variants.items():
                models = [
                    fit_one_epoch(tr, w, s, f"tactic_{key}_{year}_{weighting}_s{s}") for s in SEEDS
                ]
                p_test = predict(models, te["x"])
                p_train = predict(models, tr["x"])
                b, a = np.polyfit(p_train, tr["y"], 1)
                last = pd.DatetimeIndex(tr["target_month"]).year == year - 1
                shift = float(np.mean(tr["y"][last] - (a + b * p_train[last])))
                outputs = {
                    "raw": p_test,
                    "scale": a + b * p_test,
                    "scale+offset": a + b * p_test + shift,
                }
                fits.append(
                    {
                        "label": key,
                        "year": year,
                        "weighting": weighting,
                        "a": a,
                        "b": b,
                        "recent_shift": shift,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                    }
                )
                for cal, pred in outputs.items():
                    for group in ["overall", *STATE_NAMES]:
                        sel = (
                            np.ones(len(te["y"]), bool)
                            if group == "overall"
                            else (te["state"] == STATE_NAMES.index(group))
                        )
                        y = te["y"][sel].astype(float)
                        rows.append(
                            {
                                "label": key,
                                "year": year,
                                "weighting": weighting,
                                "calibration": cal,
                                "group": group,
                                "n": int(sel.sum()),
                                "persistence_mse": float(
                                    np.mean((te["persistence"][sel] - y) ** 2)
                                ),
                                "climatology_mse": float(np.mean(y**2)),
                            }
                            | decompose(pred[sel].astype(float), y)
                        )
                print(
                    key,
                    year,
                    weighting,
                    f"a={a:.3f} b={b:.2f} shift={shift:+.3f}",
                    {
                        r["calibration"]: round(r["skill_vs_climatology"], 3)
                        for r in rows[-15:]
                        if r["group"] == "overall"
                    },
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(METRICS / "rolling_origin.csv", index=False)
        del coarse, fine
    pd.DataFrame(fits).to_csv(METRICS / "fits.csv", index=False)
    table = pd.DataFrame(rows)
    pooled = (
        table.assign(sse=table.mse * table.n, sse_clim=table.climatology_mse * table.n)
        .groupby(["label", "weighting", "calibration", "group"])
        .agg(
            sse=("sse", "sum"),
            sse_clim=("sse_clim", "sum"),
            R2_mean=("R2", "mean"),
            r2_mean=("r2", "mean"),
            offset_mean=("offset_penalty", "mean"),
            skill_min=("skill_vs_climatology", "min"),
        )
        .assign(skill_pooled=lambda d: 1 - d.sse / d.sse_clim)
        .reset_index()
    )
    pooled.to_csv(METRICS / "summary_2021_2024.csv", index=False)
    (METRICS / "settings.json").write_text(
        json.dumps(
            {
                "test_years": TEST_YEARS,
                "seeds": SEEDS,
                "half_life_years": HALF_LIFE,
                "epochs": 1,
                "train_stride_days": TRAIN_STRIDE,
                "weight_decay": WEIGHT_DECAY,
            },
            indent=2,
        )
    )
    view = pooled[pooled.group == "overall"].sort_values("skill_pooled", ascending=False)
    print(
        view[
            [
                "label",
                "weighting",
                "calibration",
                "skill_pooled",
                "skill_min",
                "R2_mean",
                "r2_mean",
                "offset_mean",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
