"""Single-state VCI-only models vs the three-state model (see RESULTS_vci.md).

For one state: train on that state's cells only and forecast that state, with the
vci_tactics.py setup otherwise unchanged (daily VCI input, 10-40 cm soil-moisture label,
uniform weights, raw output, rolling origin 2021-2024, seeds 42-44). Each model gets the same
number of training draws as the three-state model had for that test year, so both see the
same number of optimizer steps. Run one process per state, in parallel:

    python vci_per_state.py --state kansas & python vci_per_state.py --state texas &
    python vci_per_state.py --state california & wait
"""

import argparse

import numpy as np
import pandas as pd
import torch
import train_v2_delta as D
from data_vci_daily import first_anchor, load_daily, samples_for
from train_v2 import STATE_NAMES
from train_vci_daily import TRAIN_STRIDE
from train_vci_daily_recent import decompose, window_period
from vci_tactics import SEEDS, TEST_YEARS, fit_one_epoch, layer_labels, predict, select

OUT = D.OUT
METRICS = OUT / "metrics_vci_per_state"
LABEL = "sm_10_40"


def only_state(daily, index):
    keep = daily["state"] == index
    cell_keys = ("vci", "sm", "state", "lon", "lat")
    return {k: (v[keep] if k in cell_keys else v) for k, v in daily.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, choices=STATE_NAMES)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    METRICS.mkdir(exist_ok=True)
    D.INPUTS = ["vci"]
    daily = load_daily()
    days = pd.DatetimeIndex(daily["days"])
    daily = daily | {"sm": layer_labels(days)[LABEL]}
    all_cells = len(daily["state"])
    index = STATE_NAMES.index(args.state)
    state = only_state(daily, index)
    anchors = np.arange(first_anchor(daily), len(days))
    coarse = samples_for(state, anchors[::TRAIN_STRIDE])
    fine = samples_for(state, anchors)
    rows = []
    for year in TEST_YEARS:
        train = window_period(coarse, {"t": ("2001-01-01", f"{year - 1}-12-31")}) == "t"
        test = window_period(fine, {"t": (f"{year}-01-01", f"{year}-12-31")}) == "t"
        tr, te = select(coarse, train), select(fine, test)
        draws = int(train.sum() / len(state["state"]) * all_cells)  # three-state model's count
        uniform = np.ones(len(tr["y"]))
        models = [
            fit_one_epoch(tr, uniform, seed, f"perstate_{args.state}_{year}_s{seed}", draws)
            for seed in SEEDS
        ]
        pred = predict(models, te["x"])
        y = te["y"].astype(float)
        rows.append(
            {
                "state": args.state,
                "year": year,
                "label": LABEL,
                "n_train": int(train.sum()),
                "draws": draws,
                "n_test": len(y),
                "climatology_mse": float(np.mean(y**2)),
                "persistence_mse": float(np.mean((te["persistence"] - y) ** 2)),
            }
            | decompose(pred.astype(float), y)
        )
        print(
            args.state,
            year,
            {k: round(rows[-1][k], 3) for k in ("mse", "skill_vs_climatology", "R2", "r2")},
            flush=True,
        )
        pd.DataFrame(rows).to_csv(METRICS / f"{args.state}.csv", index=False)


if __name__ == "__main__":
    main()
