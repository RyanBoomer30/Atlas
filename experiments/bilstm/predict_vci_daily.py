"""Daily inference: forecast the next-30-day root-zone soil-moisture anomaly from VCI.

Uses only VCI available on the given day (the latest complete MODIS 16-day composites)
and averages the three trained seed models (scratch_bilstm_vci_daily_seed{42,43,44}.pt).
Writes one row per kept cell; when FLDAS covers the next 30 days, the actual anomaly and
persistence are added for comparison.

    python predict_vci_daily.py --date 2024-07-15
    python predict_vci_daily.py --date 2024-07-15 --out forecast_2024-07-15.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data_vci_daily import BLOCK_DAYS, BLOCKS, LEAD_DAYS, load_daily, window_inputs
from train_v2 import STATE_NAMES
from train_v2_delta import RegularizedBiLSTM

OUT = Path(__file__).parent
SEEDS = (42, 43, 44)


def load_models():
    models = []
    for seed in SEEDS:
        model = RegularizedBiLSTM(1, 64)
        model.load_state_dict(torch.load(OUT / f"scratch_bilstm_vci_daily_seed{seed}.pt"))
        model.eval()
        models.append(model)
    return models


def predict(date, daily=None, models=None):
    daily = daily or load_daily()
    models = models or load_models()
    days = pd.DatetimeIndex(daily["days"])
    i = days.get_loc(pd.Timestamp(date))
    if i < BLOCKS * BLOCK_DAYS - 1 or not np.isfinite(daily["vci"][:, i - 359 : i + 1]).all():
        raise SystemExit(f"{date}: needs 360 days of VCI before it (first usable day + 359)")
    x = torch.from_numpy(window_inputs(daily["vci"], i))
    with torch.no_grad():
        pred = np.mean([m(x).numpy()[:, 0] for m in models], axis=0)
    out = pd.DataFrame(
        {
            "date": days[i].strftime("%Y-%m-%d"),
            "window": f"{days[i + 1].date() if i + 1 < len(days) else ''}..",
            "state": np.array(STATE_NAMES)[daily["state"]],
            "longitude": daily["lon"],
            "latitude": daily["lat"],
            "vci_last_30d": np.round(x[:, -1, 0].numpy() * 25 + 50, 1),
            "predicted_sm_anomaly": np.round(pred, 4),
        }
    )
    if i + LEAD_DAYS < len(days):
        sm = daily["sm"]
        out["window"] = f"{days[i + 1].date()}..{days[i + LEAD_DAYS].date()}"
        out["actual_sm_anomaly"] = np.round(sm[:, i + 1 : i + 1 + LEAD_DAYS].mean(axis=1), 4)
        out["persistence"] = np.round(sm[:, i + 1 - LEAD_DAYS : i + 1].mean(axis=1), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True, help="issue date, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=None, help="CSV path (default: print)")
    args = parser.parse_args()
    table = predict(args.date)
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}: {len(table)} cells")
    summary = (
        table.groupby("state")[
            [c for c in ("predicted_sm_anomaly", "actual_sm_anomaly", "persistence") if c in table]
        ]
        .mean()
        .round(3)
    )
    print(f"forecast issued {table.date[0]} for {table.window[0]} (state means):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
