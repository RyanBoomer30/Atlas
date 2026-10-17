"""Daily VCI -> 30-day root-zone soil-moisture anomaly (see SPLIT_vci.md, "Daily inference").

VCI is the only model input. A forecast can be issued on any day d:

- Input: 12 blocks of 30 days ending on d (d-359 .. d), each the mean daily VCI of the
  cell. Daily VCI is the VCI of the latest MODIS 16-day composite (Terra MOD13C1 or Aqua
  MYD13C1) that was complete by that day, so no day sees a composite that ends after it.
- Label: mean FLDAS 0-100 cm soil-moisture anomaly over days d+1 .. d+30. FLDAS is
  monthly, so each day carries its month's anomaly and the 30 days are averaged.

VCI per pixel uses the composite's own 16-day slot (sensor and start day of year):
100 (NDVI - min) / (max - min), with min/max over train years 2000-2014 only.

    python data_vci_daily.py     # writes .atlas/data/vci/daily_vci.npz
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from data_v2 import CELLS_CSV, MONTHS, STATE_NAMES, build_state
from data_vci import (
    GOOD_RELIABILITY,
    MIN_NDVI_RANGE,
    NDVI_DIR,
    OUT_DIR,
    VCI_CLIP,
    VCI_STATS_YEARS,
    cell_vci,
    rzsm_anomaly,
)

REPO = Path(__file__).resolve().parents[2]
COMPOSITE_DIR = REPO / ".atlas/data/mod13c1"
DAILY_CACHE = OUT_DIR / "daily_vci.npz"
DAILY_SUMMARY = OUT_DIR / "daily_summary.json"

BLOCK_DAYS = 30
BLOCKS = 12  # 360 days of VCI per forecast
LEAD_DAYS = 30  # forecast window d+1 .. d+30
AVAILABLE_AFTER = 16  # a 16-day composite starting on s is complete on s + 15
FIRST_DAY, LAST_DAY = pd.Timestamp("2000-02-18"), pd.Timestamp("2024-12-31")
# Forecast windows must fall inside one split block; anything straddling a gap is dropped.
SPLIT_DAYS = {
    "train": ("2001-01-01", "2014-12-31"),
    "val": ("2016-01-01", "2019-12-31"),
    "test": ("2021-01-01", "2024-12-31"),
}


def composites():
    """Composite table: sensor, start date, the day its VCI becomes usable, and file."""
    rows = []
    for path in sorted(COMPOSITE_DIR.glob("*.npz")):
        sensor, day = path.stem.split("_")
        start = pd.Timestamp(day)
        rows.append(
            {
                "sensor": sensor,
                "start": start,
                "doy": start.dayofyear,
                "available": start + pd.Timedelta(days=AVAILABLE_AFTER),
                "path": path,
            }
        )
    return pd.DataFrame(rows).sort_values(["available", "sensor"]).reset_index(drop=True)


def composite_vci(table):
    """VCI per pixel for every composite, using train-year min/max of its 16-day slot."""
    first = np.load(table.path[0])
    stack = np.full((len(table), *first["ndvi"].shape), np.nan, np.float32)
    for i, path in enumerate(table.path):
        with np.load(path) as f:
            ndvi, rel = f["ndvi"], f["reliability"]
        good = np.isin(rel, GOOD_RELIABILITY) & (ndvi != -3000)
        stack[i] = np.where(good, ndvi / 1e4, np.nan)
    years = table.start.dt.year.to_numpy()
    in_stats = (years >= VCI_STATS_YEARS[0]) & (years <= VCI_STATS_YEARS[1])
    vci = np.full_like(stack, np.nan)
    for (_sensor, _doy), group in table.groupby(["sensor", "doy"]):
        idx = group.index.to_numpy()
        ref = stack[idx[in_stats[idx]]]
        with warnings.catch_warnings():  # ocean pixels are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(ref, axis=0), np.nanmax(ref, axis=0)
        span = hi - lo
        usable = np.isfinite(span) & (span >= MIN_NDVI_RANGE)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = 100 * (stack[idx] - lo) / np.where(usable, span, np.nan)
        vci[idx] = np.clip(v, *VCI_CLIP)
    return vci


def daily_from_composites(cell_values, table, days):
    """Daily cell VCI: the latest composite usable on that day (missing values carried)."""
    filled = pd.DataFrame(cell_values.T).ffill().to_numpy().T  # (cells, composites)
    pos = np.searchsorted(table.available.to_numpy(), days.to_numpy(), side="right") - 1
    daily = np.full((len(filled), len(days)), np.nan, np.float32)
    ok = pos >= 0
    daily[:, ok] = filled[:, pos[ok]]
    return daily


def daily_labels(sm_monthly, days):
    """Each day carries its month's soil-moisture anomaly -> (cells, days)."""
    month_index = MONTHS.get_indexer(days.to_period("M"))
    return sm_monthly[:, month_index].astype(np.float32)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = composites()
    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
    grid = np.load(NDVI_DIR / "grid.npz")
    vci_px = composite_vci(table)
    kept = pd.read_csv(CELLS_CSV).query("kept")
    vci_daily, sm_daily, meta, summary = [], [], [], {}
    for state in STATE_NAMES:
        x, _spei, centers, _log, _table = build_state(state)
        cells = [(round(a, 2), round(b, 2)) for a, b in centers]
        assert len(cells) == len(kept[kept.state == state])
        values, counts = cell_vci(vci_px, grid["lat"], grid["lon"], cells)
        vci_daily.append(daily_from_composites(values, table, days))
        sm_daily.append(daily_labels(rzsm_anomaly(x)[1], days))
        meta.append(
            pd.DataFrame({"state": state, "longitude": centers[:, 0], "latitude": centers[:, 1]})
        )
        summary[state] = {
            "cells": len(cells),
            "composite_cell_values_missing": int(np.isnan(values).sum()),
            "share_missing": round(float(np.isnan(values).mean()), 4),
            "median_valid_pixels": int(np.median(counts)),
        }
        print(state, summary[state], flush=True)
    meta = pd.concat(meta, ignore_index=True)
    summary["composites"] = {s: int((table.sensor == s).sum()) for s in table.sensor.unique()}
    summary["first_usable_day"] = str(table.available.min().date())
    np.savez(
        DAILY_CACHE,
        vci=np.concatenate(vci_daily),
        sm=np.concatenate(sm_daily),
        days=days.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        state=meta.state.map(STATE_NAMES.index).to_numpy(np.int8),
        lon=meta.longitude.to_numpy(np.float32),
        lat=meta.latitude.to_numpy(np.float32),
    )
    DAILY_SUMMARY.write_text(json.dumps(summary, indent=2))
    return load_daily()


def load_daily():
    if not DAILY_CACHE.exists():
        return build()
    with np.load(DAILY_CACHE) as f:
        return {k: f[k] for k in f.files}


def window_inputs(vci_daily, day_index):
    """Input for anchor day index i: 12 mean-VCI blocks of 30 days, oldest first -> (cells, 12, 1)."""
    csum = np.concatenate([np.zeros((len(vci_daily), 1)), np.cumsum(vci_daily, axis=1)], axis=1)
    ends = day_index - BLOCK_DAYS * np.arange(BLOCKS - 1, -1, -1)  # inclusive block ends
    means = (csum[:, ends + 1] - csum[:, ends + 1 - BLOCK_DAYS]) / BLOCK_DAYS
    return ((means - 50.0) / 25.0)[:, :, None].astype(np.float32)


def first_anchor(daily):
    """Earliest anchor index whose 360 input days all have VCI."""
    valid = np.isfinite(daily["vci"]).all(axis=0)
    first_valid = int(np.argmax(valid))
    return first_valid + BLOCKS * BLOCK_DAYS - 1


def split_of_window(start, end):
    for name, (lo, hi) in SPLIT_DAYS.items():
        if lo <= start and end <= hi:
            return name
    return "gap"


def samples_for(daily, anchor_indices):
    """Samples (same keys as data_v2) for the given anchor day indices, all cells each."""
    days = pd.DatetimeIndex(daily["days"])
    vci, sm = daily["vci"], daily["sm"]
    csum_sm = np.concatenate([np.zeros((len(sm), 1)), np.cumsum(sm, axis=1)], axis=1)
    xs, ys, ps, anchors, splits = [], [], [], [], []
    for i in anchor_indices:
        if i + LEAD_DAYS >= len(days):
            continue
        xs.append(window_inputs(vci, i))
        ys.append((csum_sm[:, i + 1 + LEAD_DAYS] - csum_sm[:, i + 1]) / LEAD_DAYS)
        ps.append((csum_sm[:, i + 1] - csum_sm[:, i + 1 - LEAD_DAYS]) / LEAD_DAYS)
        start, end = days[i + 1].strftime("%Y-%m-%d"), days[i + LEAD_DAYS].strftime("%Y-%m-%d")
        anchors.append(days[i].strftime("%Y-%m-%d"))
        splits.append(split_of_window(start, end))
    n = len(daily["state"])
    k = len(anchors)
    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys).astype(np.float32),
        "persistence": np.concatenate(ps).astype(np.float32),
        "state": np.tile(daily["state"], k),
        "lon": np.tile(daily["lon"], k),
        "lat": np.tile(daily["lat"], k),
        "target_month": np.repeat(np.array(anchors, dtype="U10"), n),  # anchor day
        "split": np.repeat(np.array(splits, dtype="U5"), n),
    }


if __name__ == "__main__":
    build()
