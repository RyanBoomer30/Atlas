"""Dataset for VCI -> FLDAS root-zone soil moisture (see SPLIT_vci.md).

Same 359 kept 0.5 deg cells, time blocks and window as shared split v2; the input and
label change:

- Input: VCI from MODIS MOD13C2 monthly 0.05 deg NDVI. Per pixel and calendar month,
  VCI = 100 (NDVI - min) / (max - min), with min/max over train years 2000-2014 only.
  Good/marginal pixels only (reliability 0 or 1). Cell value = mean over valid pixels.
- Label: FLDAS 0-100 cm soil moisture (thickness-weighted 0-10, 10-40, 40-100 cm layers),
  as a standardized anomaly per cell and calendar month (1982-2014 statistics), at t+1.

    python data_vci.py      # writes .atlas/data/vci/samples_vci.npz and coverage_vci.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data_v2 import (
    CELLS_CSV,
    FLDAS_FEATURES,
    MONTHS,
    STATE_NAMES,
    build_state,
    snap,
)
from shared_split_v2 import HORIZON, SEQ_LEN, split_of
from torch.utils.data import TensorDataset

REPO = Path(__file__).resolve().parents[2]
NDVI_DIR = REPO / ".atlas/data/mod13c2"
OUT_DIR = REPO / ".atlas/data/vci"
CACHE = OUT_DIR / "samples_vci.npz"
COVERAGE_CSV = OUT_DIR / "coverage_vci.csv"
SUMMARY_JSON = OUT_DIR / "summary_vci.json"

VCI_FIRST = "2000-02"  # first MOD13C2 month
VCI_STATS_YEARS = (2000, 2014)  # VCI min/max: train-period inputs only
SM_STATS_YEARS = (1982, 2014)  # soil-moisture climatology: everything before val
GOOD_RELIABILITY = (0, 1)  # MOD13C2 pixel reliability: good, marginal
MIN_NDVI_RANGE = 0.05  # pixels whose train-year NDVI barely moves give no usable VCI
VCI_CLIP = (-50.0, 150.0)  # test months can leave the train-year min/max range
# FLDAS layers and thicknesses (m) for the 0-100 cm root-zone average.
SM_LAYERS = {"SoilMoi00_10cm_tavg": 0.1, "SoilMoi10_40cm_tavg": 0.3, "SoilMoi40_100cm_tavg": 0.6}

VCI_FEATURES = ["vci", "month_sin", "month_cos"]
SM_HISTORY_FEATURE = "rzsm_anomaly_history"


def load_ndvi():
    """NDVI (months, rows, cols) as float32 with bad pixels NaN, its months, and the grid."""
    grid = np.load(NDVI_DIR / "grid.npz")
    months = pd.period_range(VCI_FIRST, MONTHS[-1], freq="M")
    stack = np.full((len(months), len(grid["lat"]), len(grid["lon"])), np.nan, np.float32)
    for i, m in enumerate(months):
        with np.load(NDVI_DIR / f"{m}.npz") as f:
            ndvi, rel = f["ndvi"], f["reliability"]
        good = np.isin(rel, GOOD_RELIABILITY) & (ndvi != -3000)
        stack[i] = np.where(good, ndvi / 1e4, np.nan)
    return stack, months, grid["lat"], grid["lon"]


def pixel_vci(ndvi, months):
    """VCI per pixel and month from train-year min/max of the same calendar month."""
    vci = np.full_like(ndvi, np.nan)
    years, cal = months.year.to_numpy(), months.month.to_numpy()
    stats = (years >= VCI_STATS_YEARS[0]) & (years <= VCI_STATS_YEARS[1])
    for m in range(1, 13):
        sel = cal == m
        ref = ndvi[sel & stats]
        with np.errstate(all="ignore"):
            lo, hi = np.nanmin(ref, axis=0), np.nanmax(ref, axis=0)
        span = hi - lo
        usable = np.isfinite(span) & (span >= MIN_NDVI_RANGE)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = 100 * (ndvi[sel] - lo) / np.where(usable, span, np.nan)
        vci[sel] = np.clip(v, *VCI_CLIP)
    return vci


def cell_vci(vci, lat, lon, cells):
    """Mean VCI of the valid pixels inside each kept 0.5 deg cell -> (cells, months)."""
    lon_grid, lat_grid = np.meshgrid(snap(lon), snap(lat))
    index = {(lo, la): i for i, (lo, la) in enumerate(cells)}
    cell_of = np.array(
        [index.get((a, b), -1) for a, b in zip(lon_grid.ravel(), lat_grid.ravel(), strict=True)]
    )
    inside = cell_of >= 0
    out = np.full((len(cells), len(vci)), np.nan, np.float32)
    counts = np.zeros((len(cells), len(vci)), np.int32)
    for t, frame in enumerate(vci):
        flat = frame.ravel()
        ok = inside & np.isfinite(flat)
        n = np.bincount(cell_of[ok], minlength=len(cells))
        s = np.bincount(cell_of[ok], weights=flat[ok], minlength=len(cells))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, t] = np.where(n > 0, s / n, np.nan)
        counts[:, t] = n
    return out, counts


def fill_gaps(series, months):
    """Carry the last value forward (causal); leading gaps get the train-year monthly mean."""
    frame = pd.DataFrame(series.T)
    filled = frame.ffill().to_numpy().T
    years, cal = months.year.to_numpy(), months.month.to_numpy()
    stats = (years >= VCI_STATS_YEARS[0]) & (years <= VCI_STATS_YEARS[1])
    for m in range(1, 13):
        sel = cal == m
        clim = np.nanmean(series[:, sel & stats], axis=1, keepdims=True)
        filled[:, sel] = np.where(np.isnan(filled[:, sel]), clim, filled[:, sel])
    return filled


def rzsm_anomaly(x):
    """Root-zone (0-100 cm) soil moisture anomaly per cell and calendar month -> (cells, 516)."""
    depth = sum(SM_LAYERS.values())
    rz = sum(x[:, :, FLDAS_FEATURES.index(k)] * w for k, w in SM_LAYERS.items()) / depth
    years, cal = MONTHS.year.to_numpy(), MONTHS.month.to_numpy()
    stats = (years >= SM_STATS_YEARS[0]) & (years <= SM_STATS_YEARS[1])
    z = np.zeros_like(rz)
    for m in range(1, 13):
        sel = cal == m
        ref = rz[:, sel & stats]
        z[:, sel] = (rz[:, sel] - ref.mean(axis=1, keepdims=True)) / (
            ref.std(axis=1, keepdims=True) + 1e-9
        )
    return rz, z


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = pd.read_csv(CELLS_CSV).query("kept")
    ndvi, vmonths, lat, lon = load_ndvi()
    vci_px = pixel_vci(ndvi, vmonths)
    del ndvi
    parts, coverage, summary = [], [], {}
    offset = MONTHS.get_loc(vmonths[0])
    for si, state in enumerate(STATE_NAMES):
        x, _spei, centers, _log, _table = build_state(state)
        cells = [(round(a, 2), round(b, 2)) for a, b in centers]
        expected = kept[kept.state == state]
        assert len(cells) == len(expected), "kept cells differ from cells_v2.csv"
        raw, counts = cell_vci(vci_px, lat, lon, cells)
        missing = np.isnan(raw)
        vci = fill_gaps(raw, vmonths)
        _rz, sm = rzsm_anomaly(x)
        sm = sm[:, offset:]  # align to the VCI months
        summary[state] = {
            "cells": len(cells),
            "cell_months_without_valid_vci": int(missing.sum()),
            "share_filled": round(float(missing.mean()), 4),
            "median_valid_pixels_per_cell_month": int(np.median(counts)),
        }
        coverage.append(
            pd.DataFrame(
                {
                    "state": state,
                    "longitude": centers[:, 0],
                    "latitude": centers[:, 1],
                    "months_filled": missing.sum(axis=1),
                    "median_valid_pixels": np.median(counts, axis=1),
                }
            )
        )
        cal = vmonths.month.to_numpy()
        angle = 2 * np.pi * (cal - 1) / 12
        feats = np.stack(
            [
                (vci - 50.0) / 25.0,  # VCI 0..100 -> about -2..2
                np.broadcast_to(np.sin(angle), vci.shape),
                np.broadcast_to(np.cos(angle), vci.shape),
            ],
            axis=2,
        ).astype(np.float32)
        ends = np.arange(SEQ_LEN - 1, len(vmonths) - HORIZON)
        win = np.lib.stride_tricks.sliding_window_view(feats, SEQ_LEN, axis=1)[:, : len(ends)]
        hist = np.lib.stride_tricks.sliding_window_view(sm, SEQ_LEN, axis=1)[:, : len(ends)]
        target = np.array(vmonths[ends + HORIZON].strftime("%Y-%m"), dtype="U7")
        n = len(cells)
        parts.append(
            {
                "x": win.transpose(0, 1, 3, 2).reshape(-1, SEQ_LEN, feats.shape[2]),
                "sm_hist": hist.reshape(-1, SEQ_LEN).astype(np.float32),
                "y": sm[:, ends + HORIZON].reshape(-1).astype(np.float32),
                "persistence": sm[:, ends].reshape(-1).astype(np.float32),
                "state": np.full(n * len(ends), si, np.int8),
                "lon": np.repeat(centers[:, 0], len(ends)).astype(np.float32),
                "lat": np.repeat(centers[:, 1], len(ends)).astype(np.float32),
                "target_month": np.tile(target, n),
                "split": np.tile(np.array([split_of(m) or "gap" for m in target], dtype="U5"), n),
            }
        )
        print(state, summary[state])
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(CACHE, **data)
    pd.concat(coverage).to_csv(COVERAGE_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    return data


def load_samples(sm_history=False):
    """VCI samples; ``sm_history`` appends the soil-moisture anomaly at t-11..t as a channel."""
    if not CACHE.exists():
        build()
    with np.load(CACHE) as f:
        samples = {k: f[k] for k in f.files}
    if sm_history:
        samples["x"] = np.concatenate([samples["x"], samples["sm_hist"][:, :, None]], axis=2)
    return samples


def load_splits(samples):
    return {
        name: TensorDataset(
            torch.from_numpy(samples["x"][samples["split"] == name]),
            torch.from_numpy(samples["y"][samples["split"] == name]).unsqueeze(1),
        )
        for name in ("train", "val", "test")
    }


if __name__ == "__main__":
    build()
