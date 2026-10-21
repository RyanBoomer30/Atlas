"""Regional VCI-only models: California alone, Kansas + Texas together (see RESULTS_vci.md).

Setup as vci_tactics.py / vci_per_state.py: daily VCI is the only input, 10-40 cm soil-moisture
label, raw output, rolling origin 2021-2024 (train 2001 .. year before), one epoch, seeds 42-44,
and the same number of training draws as the three-state model (same optimizer steps). Within a
region, states are drawn in proportion to their effective number of independent cells.

--spatial trains without the 8 held-out 2x2 deg blocks of train_v2.spatial_holdout (chosen
over all three states, as before) and scores only the region's held-out cells.

    python vci_regional.py --region california
    python vci_regional.py --region plains            # Kansas + Texas
    python vci_regional.py --region plains --spatial
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
from vci_tactics import SEEDS, TEST_YEARS, fit_one_epoch, layer_labels, predict, select

OUT = D.OUT
METRICS = OUT / "metrics_vci_regional"
LABEL = "sm_10_40"
REGIONS = {"california": ["california"], "plains": ["kansas", "texas"]}


def cells_of(daily, states):
    keep = np.isin(daily["state"], [STATE_NAMES.index(s) for s in states])
    cell_keys = ("vci", "sm", "state", "lon", "lat")
    return {k: (v[keep] if k in cell_keys else v) for k, v in daily.items()}


def held_cells(daily):
    """(lon, lat) of cells inside the held-out blocks, chosen over all three states."""
    every = {"lon": daily["lon"], "lat": daily["lat"]}
    held, _, _ = holdout_blocks(every)
    return set(zip(daily["lon"][held].round(2), daily["lat"][held].round(2), strict=True))


def in_cells(samples, cells):
    pairs = zip(samples["lon"].round(2), samples["lat"].round(2), strict=True)
    return np.array([p in cells for p in pairs])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", choices=list(REGIONS), required=True)
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    METRICS.mkdir(exist_ok=True)
    D.INPUTS = ["vci"]
    daily = load_daily()
    days = pd.DatetimeIndex(daily["days"])
    daily = daily | {"sm": layer_labels(days)[LABEL]}
    all_cells = len(daily["state"])
    held = held_cells(daily) if args.spatial else set()
    region = cells_of(daily, REGIONS[args.region])
    anchors = np.arange(first_anchor(daily), len(days))
    coarse = samples_for(region, anchors[::TRAIN_STRIDE])
    fine = samples_for(region, anchors)

    base = window_period(coarse, {"t": ("2001-01-01", "2020-12-31")}) == "t"
    n_eff = D.effective_cells(
        select(coarse, base) | {"split": np.full(base.sum(), "train", dtype="U5")}
    )
    present = {s: n for s, n in n_eff.items() if s in REGIONS[args.region]}
    shares = {s: n / sum(present.values()) for s, n in present.items()}
    held_coarse = in_cells(coarse, held) if held else np.zeros(len(coarse["y"]), bool)
    held_fine = in_cells(fine, held) if held else np.ones(len(fine["y"]), bool)

    tag = f"{args.region}_{'spatial' if args.spatial else 'main'}"
    rows, equivalence_ok = [], None
    for year in TEST_YEARS:
        train = window_period(coarse, {"t": ("2001-01-01", f"{year - 1}-12-31")}) == "t"
        train &= ~held_coarse
        test = window_period(fine, {"t": (f"{year}-01-01", f"{year}-12-31")}) == "t"
        test &= held_fine
        tr, te = select(coarse, train), select(fine, test)
        draws = int(train.sum() / len(region["state"]) * all_cells)  # three-state model's count
        counts = np.bincount(tr["state"], minlength=len(STATE_NAMES))
        share = np.array([shares.get(s, 0.0) for s in STATE_NAMES])
        weights = share[tr["state"]] / counts[tr["state"]]
        models = [
            fit_one_epoch(tr, weights, s, f"regional_{tag}_{year}_s{s}", draws) for s in SEEDS
        ]
        if equivalence_ok is None:
            equivalence_ok = equivalence(models[0])[1]
        pred = predict(models, te["x"])
        for group in ["region", *REGIONS[args.region]]:
            sel = (
                np.ones(len(te["y"]), bool)
                if group == "region"
                else (te["state"] == STATE_NAMES.index(group))
            )
            if group == "region" and len(REGIONS[args.region]) == 1:
                continue
            y = te["y"][sel].astype(float)
            rows.append(
                {
                    "region": args.region,
                    "run": "spatial_unseen" if args.spatial else "test",
                    "year": year,
                    "group": group,
                    "n": int(sel.sum()),
                    "draws": draws,
                    "climatology_mse": float(np.mean(y**2)),
                    "persistence_mse": float(np.mean((te["persistence"][sel] - y) ** 2)),
                }
                | decompose(pred[sel].astype(float), y)
            )
        print(
            tag,
            year,
            {r["group"]: round(r["skill_vs_climatology"], 3) for r in rows if r["year"] == year},
            flush=True,
        )
        pd.DataFrame(rows).assign(equivalence_pass=equivalence_ok).to_csv(
            METRICS / f"{tag}.csv", index=False
        )


if __name__ == "__main__":
    main()
