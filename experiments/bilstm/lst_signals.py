"""Which land-surface temperature signals track soil moisture? A check before any training.

For Terra (10:30 / 22:30) and Aqua (13:30 / 01:30) MOD11C2 / MYD11C2 8-day LST, three daily
per-cell indices built like TCI (data_us.slot_index, latest composite complete by the day):

- day:   daytime LST (high = hot = low index), the TCI used so far for Terra
- night: nighttime LST, same formula
- range: day minus night LST (a wide day-night range = low index)

Each index is lined up the way the model sees it: its mean over the 30 days up to issue day d
against the mean 10-40 cm soil-moisture anomaly over d+1 .. d+30, every 5th day, train
years 2003-2014 only (Aqua starts in mid-2002; no test year is used). Per cell:

- r: correlation of the index with the label
- added r2: r2 of a per-cell linear fit label ~ VCI + Terra day TCI + index, minus r2 of
  label ~ VCI + Terra day TCI (what the index adds beyond the current inputs)

Medians over cells, for West / East of 98 W and Kansas + Texas + California.

    python lst_signals.py     # writes .atlas/data/vci_us/lst_signals.npz and metrics_vci_us/lst_signals.csv
"""

import numpy as np
import pandas as pd
from data_us import (
    CMG_COLS,
    CMG_ROWS,
    DAILY_US,
    OUT_US_DIR,
    cell_vci,
    daily_from_composites,
    load,
    load_lst,
    lst_composites,
    slot_index,
)
from plot_us import METRICS
from shared_split_us import EAST, MOD11C2_US_DIR, MYD11C2_US_DIR, THREE_STATES, WEST

SIGNALS_US = OUT_US_DIR / "lst_signals.npz"
PRODUCTS = {"terra": MOD11C2_US_DIR, "aqua": MYD11C2_US_DIR}
WINDOW = 30
STRIDE = 5
YEARS = ("2003-01-01", "2014-12-31")
GROUPS = {"west": WEST, "east": EAST, "KS+TX+CA": THREE_STATES}


def cell_series(pixel_index, table, cells, days):
    lat = 90 - 0.025 - 0.05 * np.arange(CMG_ROWS[0], CMG_ROWS[1] + 1)
    lon = -180 + 0.025 + 0.05 * np.arange(CMG_COLS[0], CMG_COLS[1] + 1)
    values, _ = cell_vci(pixel_index, lat, lon, cells)
    return daily_from_composites(values, table, days)


def build():
    with np.load(DAILY_US) as f:
        lon, lat, days = f["lon"], f["lat"], pd.DatetimeIndex(f["days"])
    cells = [(round(float(a), 2), round(float(b), 2)) for a, b in zip(lon, lat, strict=True)]
    out = {}
    for product, directory in PRODUCTS.items():
        table = lst_composites(directory)
        day = load_lst(table, "lst", "qc")
        night = load_lst(table, "lst_night", "qc_night")
        spread = day - night  # NaN unless both are valid
        for name, stack in (("day", day), ("night", night), ("range", spread)):
            out[f"{product}_{name}"] = cell_series(slot_index(stack, table), table, cells, days)
            print(product, name, f"{np.isnan(out[f'{product}_{name}']).mean():.1%} missing days")
        del day, night, spread
    np.savez(SIGNALS_US, **out)
    return out


def trailing_mean(series, anchors):
    """Mean of the finite values over the WINDOW days up to and including each anchor."""
    finite = np.isfinite(series)
    zero = np.zeros((len(series), 1))
    total = np.concatenate([zero, np.cumsum(np.where(finite, series, 0.0), axis=1)], axis=1)
    count = np.concatenate([zero, np.cumsum(finite, axis=1)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = count[:, anchors + 1] - count[:, anchors + 1 - WINDOW]
        return (total[:, anchors + 1] - total[:, anchors + 1 - WINDOW]) / n


def r2(y, columns):
    x = np.column_stack([np.ones(len(y)), *columns])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return 1 - np.var(y - x @ beta) / np.var(y)


def main():
    signals = dict(np.load(SIGNALS_US)) if SIGNALS_US.exists() else build()
    data = load("sm_10_40")
    days = pd.DatetimeIndex(data["days"])
    i = np.arange(WINDOW, len(days) - WINDOW)[::STRIDE]
    anchors = i[(days[i + 1] >= YEARS[0]) & (days[i + WINDOW] <= YEARS[1])]
    sm = data["sm"]
    csum = np.concatenate([np.zeros((len(sm), 1)), np.cumsum(sm, axis=1)], axis=1)
    label = (csum[:, anchors + 1 + WINDOW] - csum[:, anchors + 1]) / WINDOW
    vci, tci = trailing_mean(data["vci"], anchors), trailing_mean(data["tci"], anchors)
    candidates = {"vci": vci, "terra_day (current TCI)": tci} | {
        k: trailing_mean(v, anchors) for k, v in signals.items() if k != "terra_day"
    }
    states = data["names"][data["state"]]
    rows = []
    for c in range(len(label)):
        y = label[c]
        base_ok = np.isfinite(y) & np.isfinite(vci[c]) & np.isfinite(tci[c])
        for name, values in candidates.items():
            ok = base_ok & np.isfinite(values[c])
            if ok.sum() < 100:
                continue
            base = r2(y[ok], [vci[c][ok], tci[c][ok]])
            extra = values[c] if name not in ("vci", "terra_day (current TCI)") else None
            rows.append(
                {
                    "state": states[c],
                    "signal": name,
                    "r": np.corrcoef(values[c][ok], y[ok])[0, 1],
                    "added_r2": np.nan
                    if extra is None
                    else r2(y[ok], [vci[c][ok], tci[c][ok], extra[ok]]) - base,
                    "n": int(ok.sum()),
                }
            )
    table = pd.DataFrame(rows)
    summary = []
    for group, members in GROUPS.items():
        sub = table[table.state.isin(members)]
        g = sub.groupby("signal").agg(
            r=("r", "median"), added_r2=("added_r2", "median"), cells=("r", "size")
        )
        summary.append(g.assign(group=group).reset_index())
    summary = pd.concat(summary)
    summary.to_csv(METRICS / "lst_signals.csv", index=False)
    order = list(candidates)
    for metric in ("r", "added_r2"):
        view = summary.pivot(index="signal", columns="group", values=metric).loc[
            order, list(GROUPS)
        ]
        print(f"\nmedian per-cell {metric}")
        print(view.round(3).to_string())


if __name__ == "__main__":
    main()
