"""VCI plus coordinates as input, vs VCI only (see RESULTS_vci.md).

Input: the daily VCI blocks of data_vci_daily.py plus the cell's latitude and longitude,
standardized over the 359 kept cells and repeated at every time step -> (12, 3). Everything
else is the vci_tactics.py setup: three-state model, uniform weights, raw output, rolling
origin 2021-2024 (train 2001 .. year before), one epoch, seeds 42-44.

--spatial trains without the 8 held-out 2x2 deg blocks of train_v2.spatial_holdout and
scores only those cells, to see whether coordinates help in places the model never saw.

    python vci_coords.py --inputs coords --label sm_10_40
    python vci_coords.py --inputs coords --label sm_10_40 --spatial
    python vci_coords.py --inputs vci --label sm_10_40 --spatial
"""

import argparse

import numpy as np
import pandas as pd
import torch
import train_v2_delta as D
from data_vci_daily import first_anchor, load_daily, samples_for
from train_v2 import STATE_NAMES, equivalence
from train_v2 import spatial_holdout as holdout_blocks
from train_vci_daily import TRAIN_STRIDE
from train_vci_daily_recent import decompose, window_period
from vci_tactics import (
    LAYERS,
    SEEDS,
    TEST_YEARS,
    fit_one_epoch,
    layer_labels,
    predict,
    select,
    state_weights,
)

OUT = D.OUT
METRICS = OUT / "metrics_vci_coords"


def add_coordinates(samples, lat_stats, lon_stats):
    """Append standardized latitude and longitude as constant channels."""
    n, steps, _ = samples["x"].shape
    lat = (samples["lat"] - lat_stats[0]) / lat_stats[1]
    lon = (samples["lon"] - lon_stats[0]) / lon_stats[1]
    coords = np.broadcast_to(np.stack([lat, lon], axis=1)[:, None, :], (n, steps, 2))
    return samples | {"x": np.concatenate([samples["x"], coords], axis=2).astype(np.float32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", choices=["vci", "coords"], required=True)
    parser.add_argument("--label", choices=list(LAYERS), default="sm_10_40")
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    METRICS.mkdir(exist_ok=True)
    daily = load_daily()
    days = pd.DatetimeIndex(daily["days"])
    daily = daily | {"sm": layer_labels(days)[args.label]}
    anchors = np.arange(first_anchor(daily), len(days))
    coarse = samples_for(daily, anchors[::TRAIN_STRIDE])
    fine = samples_for(daily, anchors)
    if args.inputs == "coords":
        lat_stats = (float(daily["lat"].mean()), float(daily["lat"].std()))
        lon_stats = (float(daily["lon"].mean()), float(daily["lon"].std()))
        coarse = add_coordinates(coarse, lat_stats, lon_stats)
        fine = add_coordinates(fine, lat_stats, lon_stats)
    D.INPUTS = ["vci"] + (["lat", "lon"] if args.inputs == "coords" else [])

    base_train = window_period(coarse, {"t": ("2001-01-01", "2020-12-31")}) == "t"
    n_eff = D.effective_cells(
        select(coarse, base_train) | {"split": np.full(base_train.sum(), "train", dtype="U5")}
    )
    shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}
    held_coarse = holdout_blocks(coarse)[0] if args.spatial else np.zeros(len(coarse["y"]), bool)
    held_fine = holdout_blocks(fine)[0] if args.spatial else np.ones(len(fine["y"]), bool)

    tag = f"{args.inputs}_{args.label}_{'spatial' if args.spatial else 'main'}"
    rows, equivalence_ok = [], None
    for year in TEST_YEARS:
        train = window_period(coarse, {"t": ("2001-01-01", f"{year - 1}-12-31")}) == "t"
        train &= ~held_coarse
        test = window_period(fine, {"t": (f"{year}-01-01", f"{year}-12-31")}) == "t"
        test &= held_fine
        tr, te = select(coarse, train), select(fine, test)
        weights = state_weights(tr["state"], shares)
        models = [fit_one_epoch(tr, weights, s, f"coords_{tag}_{year}_s{s}") for s in SEEDS]
        if equivalence_ok is None:
            equivalence_ok = equivalence(models[0])[1]
        pred = predict(models, te["x"])
        for group in ["overall", *STATE_NAMES]:
            sel = (
                np.ones(len(te["y"]), bool)
                if group == "overall"
                else (te["state"] == STATE_NAMES.index(group))
            )
            y = te["y"][sel].astype(float)
            rows.append(
                {
                    "inputs": args.inputs,
                    "label": args.label,
                    "run": "spatial_unseen" if args.spatial else "test",
                    "year": year,
                    "group": group,
                    "n": int(sel.sum()),
                    "climatology_mse": float(np.mean(y**2)),
                    "persistence_mse": float(np.mean((te["persistence"][sel] - y) ** 2)),
                }
                | decompose(pred[sel].astype(float), y)
            )
        overall = rows[-1 - len(STATE_NAMES)]
        print(
            tag,
            year,
            {k: round(overall[k], 3) for k in ("skill_vs_climatology", "r2", "R2")},
            flush=True,
        )
        pd.DataFrame(rows).assign(equivalence_pass=equivalence_ok).to_csv(
            METRICS / f"{tag}.csv", index=False
        )


if __name__ == "__main__":
    main()
