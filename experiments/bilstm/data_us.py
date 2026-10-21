"""Contiguous-US daily VCI and FLDAS soil-moisture labels on 0.5 deg cells (shared_split_us.py).

The cell rules are data_v2.build_state's, applied to every contiguous state at once:
SPEI-12 present, cell center inside a state border, FLDAS present in every month, not
hyper-arid, stable SPEI-12, and >= 30 % grassland/cropland/mosaic (MCD12Q1 2010). The input
and labels are data_vci_daily's and vci_tactics.layer_labels': daily VCI from the latest
complete MOD13C1/MYD13C1 composite, and daily FLDAS soil-moisture anomalies per layer.

Differences from the three-state build: FLDAS is pulled for the whole box (no state box cuts
a border cell), and only the variables the VCI setup uses are pulled (soil layers and rain),
so the "FLDAS present" rule checks those four.

    python data_us.py    # writes .atlas/data/vci_us/{daily_vci.npz, labels.npz, cells.csv, masks.json}
    python data_us.py --tci   # adds daily_tci.npz (needs daily_vci.npz and download_lst8.py)

TCI (temperature condition index) mirrors VCI for MOD11C2 8-day daytime land-surface
temperature: 100 (LSTmax - LST) / (LSTmax - LSTmin) per pixel and 8-day slot, min/max over
train years 2000-2014, so 0 is the hottest (most stressed) and 100 the coolest. Daily TCI is
the latest composite complete by that day (an 8-day composite starting on s is usable on s + 8).
"""

import argparse
import json
import warnings
from itertools import pairwise

import numpy as np
import pandas as pd
import shapefile
import xarray as xr
from data_v2 import (
    CENSUS_SHP,
    MONTHS,
    SPEI_DIR,
    STUCK_RUN,
    inside,
    land_cover_fraction,
    longest_stuck_run,
    snap,
)
from data_vci import SM_LAYERS, SM_STATS_YEARS, VCI_CLIP, VCI_STATS_YEARS, cell_vci
from data_vci_daily import (
    AVAILABLE_AFTER,
    FIRST_DAY,
    LAST_DAY,
    composite_vci,
    daily_from_composites,
    daily_labels,
)
from shared_split_us import (
    BOX,
    CMG_COLS,
    CMG_ROWS,
    EXCLUDED_FIPS,
    FLDAS_US_DIR,
    FLDAS_US_VARIABLES,
    MOD11C2_US_DIR,
    MOD13C1_US_DIR,
    OUT_US_DIR,
)
from shared_split_v2 import (
    ARID_MAX_FRACTION,
    ARID_RAIN_MM,
    LAND_MIN_FRACTION,
    LAND_YEAR,
    SPEI_ABS_MAX,
    STATS_YEARS,
)

DAILY_US = OUT_US_DIR / "daily_vci.npz"
LABELS_US = OUT_US_DIR / "labels.npz"
TCI_US = OUT_US_DIR / "daily_tci.npz"
LST_GOOD_QC = (0, 1)  # MOD11C2 QC bits 0-1: LST produced, good / other quality
LST_AVAILABLE_AFTER = 8
MIN_LST_RANGE = 2.0  # K: pixels whose train-year LST barely moves give no usable TCI
CELLS_US = OUT_US_DIR / "cells.csv"
MASKS_US = OUT_US_DIR / "masks.json"
LAYERS = {
    "sm_0_10": ["SoilMoi00_10cm_tavg"],
    "sm_10_40": ["SoilMoi10_40cm_tavg"],
    "sm_40_100": ["SoilMoi40_100cm_tavg"],
    "sm_0_100": list(SM_LAYERS),
}


def us_states():
    """{name: polygons} for every contiguous state (and DC), names like 'north_dakota'."""
    from matplotlib.path import Path as Polygon

    reader = shapefile.Reader(str(CENSUS_SHP))
    out = {}
    for record, shape in zip(reader.records(), reader.shapes(), strict=True):
        if record["STATEFP"] in EXCLUDED_FIPS:
            continue
        bounds = [*shape.parts, len(shape.points)]
        name = record["NAME"].lower().replace(" ", "_")
        out[name] = [Polygon(shape.points[a:b]) for a, b in pairwise(bounds)]
    return dict(sorted(out.items()))


def aggregate_tile(tile_dir):
    """Mean of the valid 0.1 deg cells per 0.5 deg cell and month -> cells, (cells, months, F)."""
    dtypes = {name: "float32" for name in FLDAS_US_VARIABLES}
    frame = pd.concat(
        pd.read_csv(p, dtype=dtypes) for p in sorted(tile_dir.glob("**/fldas.csv"))
    ).drop_duplicates(["longitude", "latitude", "time"])
    frame = frame.dropna(subset=FLDAS_US_VARIABLES)
    frame = frame.assign(
        cell_lon=snap(frame["longitude"].to_numpy()),
        cell_lat=snap(frame["latitude"].to_numpy()),
        month=pd.PeriodIndex(frame["time"].str[:7], freq="M"),
    )
    means = frame.groupby(["cell_lon", "cell_lat", "month"])[FLDAS_US_VARIABLES].mean()
    cells = sorted({(lon, lat) for lon, lat, _ in means.index})
    if not cells:  # all-ocean tile
        return [], np.zeros((0, len(MONTHS), len(FLDAS_US_VARIABLES)), np.float32)
    full = pd.MultiIndex.from_tuples([(lon, lat, m) for lon, lat in cells for m in MONTHS])
    values = means.reindex(full).to_numpy(np.float32).reshape(len(cells), len(MONTHS), -1)
    return cells, values


def load_fldas():
    cells, values = [], []
    for tile in sorted(p for p in FLDAS_US_DIR.iterdir() if p.is_dir()):
        c, v = aggregate_tile(tile)
        cells += c
        values.append(v)
        print(f"FLDAS tile {tile.name}: {len(c)} cells", flush=True)
    assert len(set(cells)) == len(cells), "a 0.5 deg cell appears in two tiles"
    return cells, np.concatenate(values)


def load_spei(cells):
    """SPEI-12 (cells, months) from the cached global SPEIbase file."""
    ds = xr.open_dataset(SPEI_DIR / "spei12.nc", engine="h5netcdf")
    west, south, east, north = BOX
    box = ds["spei"].sel(
        lon=ds.lon[(ds.lon > west) & (ds.lon < east)],
        lat=ds.lat[(ds.lat > south) & (ds.lat < north)],
        time=slice(str(MONTHS[0]), str(MONTHS[-1].end_time.date())),
    )
    frame = box.to_dataframe(name="spei").reset_index()
    frame["month"] = pd.PeriodIndex(frame["time"].dt.strftime("%Y-%m"), freq="M")
    frame["lon"], frame["lat"] = frame["lon"].round(2), frame["lat"].round(2)
    series = frame.set_index(["lon", "lat", "month"])["spei"]
    full = pd.MultiIndex.from_tuples([(lon, lat, m) for lon, lat in cells for m in MONTHS])
    return series.reindex(full).to_numpy(np.float64).reshape(len(cells), len(MONTHS))


def build_cells():
    """Kept cells, their state and FLDAS values, plus the mask log and the full cell table."""
    cells, x = load_fldas()
    spei = load_spei(cells)
    centers = np.array(cells)
    states = us_states()
    names = list(states)
    state = np.full(len(cells), -1)
    for i, polygons in enumerate(states.values()):
        state[(state < 0) & inside(polygons, centers)] = i
    keep = np.ones(len(cells), bool)
    log = [{"rule": "0.5 deg cells with any valid 0.1 deg FLDAS land cell", "kept": len(cells)}]

    def apply(rule, ok):
        nonlocal keep
        removed = int((keep & ~ok).sum())
        keep &= ok
        log.append({"rule": rule, "removed": removed, "kept": int(keep.sum())})
        print(log[-1], flush=True)

    apply("SPEI-12 present for the cell (non-ocean in SPEIbase)", np.isfinite(spei).any(axis=1))
    apply("cell center inside a contiguous-US state border (Census cb_2023 20m)", state >= 0)
    apply("FLDAS soil layers and rain present in every month", np.isfinite(x).all(axis=(1, 2)))
    years = MONTHS.year
    stats = (years >= STATS_YEARS[0]) & (years <= STATS_YEARS[1])
    days = MONTHS.days_in_month.to_numpy()
    rain_mm = x[:, :, FLDAS_US_VARIABLES.index("Rainf_f_tavg")] * 86400 * days
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
            "state": [names[s] if s >= 0 else "" for s in state],
            "longitude": centers[:, 0],
            "latitude": centers[:, 1],
            "arid_share": arid_share,
            "grass_crop_share": land,
            "longest_stuck_run": stuck,
            "kept": keep,
        }
    )
    return names, state[keep], centers[keep], x[keep], log, table


def layer_anomalies(x, days):
    """Daily soil-moisture anomaly per label key -> {key: (cells, days)}."""
    years, cal = MONTHS.year.to_numpy(), MONTHS.month.to_numpy()
    stats = (years >= SM_STATS_YEARS[0]) & (years <= SM_STATS_YEARS[1])
    out = {}
    for key, layers in LAYERS.items():
        w = np.array([SM_LAYERS[n] for n in layers])
        v = sum(x[:, :, FLDAS_US_VARIABLES.index(n)] * wi for n, wi in zip(layers, w, strict=True))
        v = v / w.sum()
        z = np.zeros_like(v)
        for m in range(1, 13):
            sel = cal == m
            ref = v[:, sel & stats]
            z[:, sel] = (v[:, sel] - ref.mean(1, keepdims=True)) / (
                ref.std(1, keepdims=True) + 1e-9
            )
        out[key] = daily_labels(z, days)
    return out


def composites():
    rows = []
    for path in sorted(MOD13C1_US_DIR.glob("*.npz")):
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


def build():
    OUT_US_DIR.mkdir(parents=True, exist_ok=True)
    names, state, centers, x, log, table = build_cells()
    table.to_csv(CELLS_US, index=False)
    per_state = pd.Series(np.array(names)[state]).value_counts().to_dict()
    MASKS_US.write_text(json.dumps({"rules": log, "kept_per_state": per_state}, indent=2))
    days = pd.date_range(FIRST_DAY, LAST_DAY, freq="D")
    np.savez(LABELS_US, **layer_anomalies(x, days))
    del x
    table_c = composites()
    vci_px = composite_vci(table_c)
    lat = 90 - 0.025 - 0.05 * np.arange(CMG_ROWS[0], CMG_ROWS[1] + 1)
    lon = -180 + 0.025 + 0.05 * np.arange(CMG_COLS[0], CMG_COLS[1] + 1)
    cells = [(round(a, 2), round(b, 2)) for a, b in centers]
    values, counts = cell_vci(vci_px, lat, lon, cells)
    del vci_px
    np.savez(
        DAILY_US,
        vci=daily_from_composites(values, table_c, days),
        days=days.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        state=state.astype(np.int8),
        lon=centers[:, 0].astype(np.float32),
        lat=centers[:, 1].astype(np.float32),
        names=np.array(names),
        median_valid_pixels=np.median(counts, axis=1),
    )
    print(f"{len(cells)} cells in {len(per_state)} states, {len(table_c)} composites", flush=True)


def lst_composites():
    rows = []
    for path in sorted(MOD11C2_US_DIR.glob("*.npz")):
        start = pd.Timestamp(path.stem.split("_")[1])
        rows.append(
            {
                "start": start,
                "doy": start.dayofyear,
                "available": start + pd.Timedelta(days=LST_AVAILABLE_AFTER),
                "path": path,
            }
        )
    return pd.DataFrame(rows).sort_values("available").reset_index(drop=True)


def composite_tci(table):
    """TCI per pixel for every composite (computed in place over one LST stack)."""
    first = np.load(table.path[0])["lst"]
    stack = np.full((len(table), *first.shape), np.nan, np.float32)
    for i, path in enumerate(table.path):
        with np.load(path) as f:
            lst, qc = f["lst"], f["qc"]
        good = np.isin(qc & 3, LST_GOOD_QC) & (lst > 0)
        stack[i] = np.where(good, lst * 0.02, np.nan)
    years = table.start.dt.year.to_numpy()
    in_stats = (years >= VCI_STATS_YEARS[0]) & (years <= VCI_STATS_YEARS[1])
    for _doy, group in table.groupby("doy"):
        idx = group.index.to_numpy()
        ref = stack[idx[in_stats[idx]]]
        with warnings.catch_warnings():  # ocean / never-clear pixels are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(ref, axis=0), np.nanmax(ref, axis=0)
        span = hi - lo
        usable = np.isfinite(span) & (span >= MIN_LST_RANGE)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = 100 * (hi - stack[idx]) / np.where(usable, span, np.nan)
        stack[idx] = np.clip(t, *VCI_CLIP)
    return stack


def build_tci():
    with np.load(DAILY_US) as f:
        lon, lat, days = f["lon"], f["lat"], pd.DatetimeIndex(f["days"])
    table = lst_composites()
    tci_px = composite_tci(table)
    grid_lat = 90 - 0.025 - 0.05 * np.arange(CMG_ROWS[0], CMG_ROWS[1] + 1)
    grid_lon = -180 + 0.025 + 0.05 * np.arange(CMG_COLS[0], CMG_COLS[1] + 1)
    cells = [(round(float(a), 2), round(float(b), 2)) for a, b in zip(lon, lat, strict=True)]
    values, counts = cell_vci(tci_px, grid_lat, grid_lon, cells)
    del tci_px
    daily = daily_from_composites(values, table, days)
    np.savez(TCI_US, tci=daily, median_valid_pixels=np.median(counts, axis=1))
    print(
        f"TCI: {len(table)} composites, {len(cells)} cells, composite cell values missing "
        f"{np.isnan(values).mean():.2%}, cells never valid {int(np.isnan(values).all(1).sum())}",
        flush=True,
    )


def load(label="sm_10_40"):
    """Daily US data with the chosen soil-moisture label as ``sm`` (built on first use).

    Adds daily ``tci`` when data_us.py --tci has been run.
    """
    if not DAILY_US.exists():
        build()
    with np.load(DAILY_US) as f, np.load(LABELS_US) as labels:
        out = {k: f[k] for k in f.files} | {"sm": labels[label]}
    if TCI_US.exists():
        with np.load(TCI_US) as f:
            out["tci"] = f["tci"]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tci", action="store_true", help="only build daily TCI")
    if parser.parse_args().tci:
        build_tci()
    else:
        build()
