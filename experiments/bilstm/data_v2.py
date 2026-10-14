"""Dataset for shared split v2 (see shared_split_v2.py and SPLIT_v2.md).

One sample stream per 0.5 deg SPEI cell. Input: the mean of every valid 0.1 deg FLDAS
cell inside the SPEI cell, z-scored per cell and calendar month with train-year
statistics, plus month-of-year sin/cos. Target: SPEI-12 at t+1, unscaled.

Build the inputs (see SPLIT_v2.md for the exact commands), then:
    python data_v2.py        # writes .atlas/data/fldas_v2/samples_v2.npz and masks_v2.json
"""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import torch
from matplotlib.path import Path as Polygon
from shared_split_v2 import (
    ARID_MAX_FRACTION,
    ARID_RAIN_MM,
    FIRST_MONTH,
    LAND_CLASSES,
    LAND_MIN_FRACTION,
    LAND_YEAR,
    LAST_MONTH,
    SEQ_LEN,
    SPEI_ABS_MAX,
    SPEI_CELL,
    STATES,
    STATS_YEARS,
    split_of,
)
from torch.utils.data import TensorDataset

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / ".atlas/data"
FLDAS_DIR = DATA / "fldas_v2"
SPEI_DIR = DATA / "spei"
CENSUS_SHP = DATA / "census/cb_2023_us_state_20m.shp"
LAND_DIR = DATA / "mcd12q1" / str(LAND_YEAR)
CACHE = FLDAS_DIR / "samples_v2.npz"
MASKS_JSON = FLDAS_DIR / "masks_v2.json"
CELLS_CSV = FLDAS_DIR / "cells_v2.csv"

FLDAS_FEATURES = [
    "SoilMoi00_10cm_tavg",
    "SoilMoi10_40cm_tavg",
    "SoilMoi40_100cm_tavg",
    "SoilMoi100_200cm_tavg",
    "Evap_tavg",
    "Qs_tavg",
    "Qsb_tavg",
    "SWE_inst",
    "SnowDepth_inst",
    "RadT_tavg",
    "Rainf_f_tavg",
    "Tair_f_tavg",
]
FEATURES = [*FLDAS_FEATURES, "month_sin", "month_cos"]
TARGET = "spei12"
STATE_NAMES = list(STATES)
MONTHS = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")
STD_FLOOR = 0.1  # per-cell std floor, as a share of the pooled std (see normalize)
Z_CLIP = 10.0
STUCK_RUN = 6  # this many identical consecutive SPEI-12 values marks an unstable fit


def snap(values):
    return np.round(np.floor(values / SPEI_CELL) * SPEI_CELL + SPEI_CELL / 2, 2)


def load_fldas(state):
    """Every 0.1 deg row of a state's pull (date chunks plus refetched months)."""
    paths = sorted((FLDAS_DIR / state).glob("**/fldas.csv"))
    dtypes = {name: "float32" for name in FLDAS_FEATURES}
    frame = pd.concat(pd.read_csv(p, dtype=dtypes) for p in paths)
    return frame.drop_duplicates(["longitude", "latitude", "time"])


def aggregate(fldas):
    """Mean of the valid 0.1 deg cells in each SPEI cell and month -> (cells, months, F)."""
    fldas = fldas.dropna(subset=FLDAS_FEATURES)  # ocean / no-data cells
    fldas = fldas.assign(
        cell_lon=snap(fldas["longitude"].to_numpy()),
        cell_lat=snap(fldas["latitude"].to_numpy()),
        month=pd.PeriodIndex(fldas["time"].str[:7], freq="M"),
    )
    grouped = fldas.groupby(["cell_lon", "cell_lat", "month"])
    means = grouped[FLDAS_FEATURES].mean()
    counts = grouped.size().groupby(["cell_lon", "cell_lat"]).max()
    cells = list(counts.index)
    full = pd.MultiIndex.from_tuples(
        [(lon, lat, m) for lon, lat in cells for m in MONTHS],
        names=["cell_lon", "cell_lat", "month"],
    )
    values = means.reindex(full).to_numpy(np.float32).reshape(len(cells), len(MONTHS), -1)
    return cells, values, counts.to_numpy()


def load_spei(state, cells):
    spei = pd.read_csv(SPEI_DIR / f"{state}_spei12.csv")
    spei["month"] = pd.PeriodIndex(spei["time"].str[:7], freq="M")
    spei = spei.set_index(["longitude", "latitude", "month"])[TARGET]
    full = pd.MultiIndex.from_tuples([(lon, lat, m) for lon, lat in cells for m in MONTHS])
    return spei.reindex(full).to_numpy(np.float64).reshape(len(cells), len(MONTHS))


def state_polygons(fips):
    """Each ring of the state's Census outline as a matplotlib Path."""
    reader = shapefile.Reader(str(CENSUS_SHP))
    field = [f[0] for f in reader.fields[1:]].index("STATEFP")
    for record, shape in zip(reader.records(), reader.shapes(), strict=True):
        if record[field] == fips:
            bounds = [*shape.parts, len(shape.points)]
            return [Polygon(shape.points[a:b]) for a, b in pairwise(bounds)]
    raise KeyError(fips)


def inside(polygons, points):
    """Even-odd rule over all rings, so holes are handled."""
    hits = sum(p.contains_points(points).astype(int) for p in polygons)
    return hits % 2 == 1


def land_cover_fraction(cells):
    """Share of MCD12Q1 land pixels in each SPEI cell that are grass/crop (NaN if none)."""
    from pyhdf.SD import SD, SDC

    radius, size, n = 6371007.181, 1111950.5197665, 2400
    pixel = size / n
    wanted = {cell: i for i, cell in enumerate(cells)}
    good = np.zeros(len(cells))
    land = np.zeros(len(cells))
    for path in sorted(LAND_DIR.glob("MCD12Q1*.hdf")):
        h, v = (
            int(part[1:3]) for part in (path.name.split(".")[2][:3], path.name.split(".")[2][3:])
        )
        lc = SD(str(path), SDC.READ).select("LC_Type1")[:]
        rows, cols = np.mgrid[0:n, 0:n]
        x = -18 * size + h * size + (cols + 0.5) * pixel
        y = 9 * size - v * size - (rows + 0.5) * pixel
        lat = np.degrees(y / radius)
        lon = np.degrees(x / (radius * np.cos(np.radians(lat))))
        is_land = (lc != 17) & (lc != 255) & (np.abs(lon) <= 180)
        keys = pd.MultiIndex.from_arrays(
            [snap(lon[is_land]), snap(lat[is_land])], names=["lon", "lat"]
        )
        counts = (
            pd.DataFrame({"good": np.isin(lc[is_land], LAND_CLASSES), "land": 1}, index=keys)
            .groupby(level=[0, 1])
            .sum()
        )
        for (lon_c, lat_c), row in counts.iterrows():
            i = wanted.get((lon_c, lat_c))
            if i is not None:
                good[i] += row["good"]
                land[i] += row["land"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(land > 0, good / land, np.nan)


def longest_stuck_run(series):
    run = best = 1
    for a, b in pairwise(series):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def build_state(state):
    info = STATES[state]
    cells, x, n_fldas = aggregate(load_fldas(state))
    spei = load_spei(state, cells)
    centers = np.array(cells)
    log = [{"rule": "SPEI cells with any valid 0.1 deg FLDAS land cell", "kept": len(cells)}]
    keep = np.ones(len(cells), bool)

    def apply(rule, ok):
        nonlocal keep
        removed = int((keep & ~ok).sum())
        keep &= ok
        log.append({"rule": rule, "removed": removed, "kept": int(keep.sum())})

    apply(
        "SPEI-12 present for the cell (non-ocean in SPEIbase)",
        np.isfinite(spei).any(axis=1),
    )
    apply(
        "cell center inside the state border (Census cb_2023 20m)",
        inside(state_polygons(info["fips"]), centers),
    )
    apply("FLDAS input present in every month 1982-01..2024-12", np.isfinite(x).all(axis=(1, 2)))

    years = MONTHS.year
    stats = (years >= STATS_YEARS[0]) & (years <= STATS_YEARS[1])
    days = MONTHS.days_in_month.to_numpy()
    rain_mm = x[:, :, FLDAS_FEATURES.index("Rainf_f_tavg")] * 86400 * days
    arid_share = (rain_mm[:, stats] < ARID_RAIN_MM).mean(axis=1)
    apply(
        f"hyper-arid: <= {ARID_MAX_FRACTION:.0%} of 1982-2014 months below {ARID_RAIN_MM} mm rain",
        arid_share <= ARID_MAX_FRACTION,
    )
    finite = np.isfinite(spei).all(axis=1)
    bounded = np.where(finite, np.nan_to_num(np.abs(spei)).max(axis=1) <= SPEI_ABS_MAX, False)
    stuck = np.array([longest_stuck_run(s) for s in spei])
    apply(
        f"stable SPEI-12: no NaN/inf, |SPEI| <= {SPEI_ABS_MAX}, no run of "
        f">= {STUCK_RUN} identical months",
        finite & bounded & (stuck < STUCK_RUN),
    )
    land = land_cover_fraction(cells)
    apply(
        f"MCD12Q1 {LAND_YEAR} grassland/cropland/mosaic >= {LAND_MIN_FRACTION:.0%} of land",
        np.nan_to_num(land, nan=0.0) >= LAND_MIN_FRACTION,
    )
    table = pd.DataFrame(
        {
            "state": state,
            "longitude": centers[:, 0],
            "latitude": centers[:, 1],
            "n_fldas_cells": n_fldas,
            "arid_share": arid_share,
            "grass_crop_share": land,
            "longest_stuck_run": stuck,
            "kept": keep,
        }
    )
    return x[keep], spei[keep], centers[keep], log, table


def normalize(x):
    """Z-score per cell, calendar month and feature using STATS_YEARS; add month sin/cos.

    A cell and month whose feature barely moves in the train years (baseflow in a dry
    cell, snow in the south) would turn any later change into a huge z. Each std is
    floored at STD_FLOOR x the feature's pooled train-year std over the state's kept
    cells, and z is clipped to +/-Z_CLIP.
    """
    years, month = MONTHS.year.to_numpy(), MONTHS.month.to_numpy()
    stats = (years >= STATS_YEARS[0]) & (years <= STATS_YEARS[1])
    floor = STD_FLOOR * x[:, stats].reshape(-1, x.shape[2]).std(axis=0)
    z = np.zeros_like(x)
    for m in range(1, 13):
        cols = month == m
        ref = x[:, cols & stats]
        mean, std = ref.mean(axis=1, keepdims=True), ref.std(axis=1, keepdims=True)
        z[:, cols] = np.clip((x[:, cols] - mean) / np.maximum(std, floor), -Z_CLIP, Z_CLIP)
    angle = 2 * np.pi * (month - 1) / 12
    season = np.broadcast_to(
        np.stack([np.sin(angle), np.cos(angle)], axis=1), (len(x), len(MONTHS), 2)
    )
    return np.concatenate([z, season], axis=2).astype(np.float32)


def windows(features, spei, centers, state_index):
    """Every (12-month input, next-month SPEI-12) pair for the kept cells."""
    ends = np.arange(SEQ_LEN - 1, len(MONTHS) - 1)
    view = np.lib.stride_tricks.sliding_window_view(features, SEQ_LEN, axis=1)
    x = view[:, : len(ends)].transpose(0, 1, 3, 2)  # (cells, samples, seq, F)
    target = np.array(MONTHS[ends + 1].strftime("%Y-%m"), dtype="U7")
    splits = np.array([split_of(m) or "gap" for m in target], dtype="U5")
    n_cells = len(features)
    return {
        "x": x.reshape(-1, SEQ_LEN, features.shape[2]),
        "y": spei[:, ends + 1].reshape(-1).astype(np.float32),
        "persistence": spei[:, ends].reshape(-1).astype(np.float32),
        "state": np.full(n_cells * len(ends), state_index, np.int8),
        "lon": np.repeat(centers[:, 0], len(ends)).astype(np.float32),
        "lat": np.repeat(centers[:, 1], len(ends)).astype(np.float32),
        "target_month": np.tile(target, n_cells),
        "split": np.tile(splits, n_cells),
    }


def build():
    parts, masks, tables = [], {}, []
    for i, state in enumerate(STATE_NAMES):
        x, spei, centers, log, table = build_state(state)
        parts.append(windows(normalize(x), spei, centers, i))
        masks[state] = log
        tables.append(table)
        print(state, json.dumps(log, indent=1))
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    np.savez(CACHE, **data)
    MASKS_JSON.write_text(json.dumps(masks, indent=2))
    pd.concat(tables).to_csv(CELLS_CSV, index=False)
    return data


def load_samples():
    """All v2 samples as numpy arrays (built and cached on first use)."""
    if not CACHE.exists():
        build()
    with np.load(CACHE) as data:
        return {key: data[key] for key in data.files}


def load_splits(samples=None, exclude=None):
    """TensorDatasets per split. ``exclude`` is a boolean mask of samples to leave out."""
    samples = samples or load_samples()
    out = {}
    for name in ("train", "val", "test"):
        keep = samples["split"] == name
        if exclude is not None:
            keep &= ~exclude
        out[name] = TensorDataset(
            torch.from_numpy(samples["x"][keep]),
            torch.from_numpy(samples["y"][keep]).unsqueeze(1),
        )
    return out


if __name__ == "__main__":
    build()
