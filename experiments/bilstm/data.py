"""Dataset for the shared split (see shared_split.py).

Features: FLDAS monthly land-surface variables per 0.1 deg cell, z-scored with
train-split statistics only. Target: SPEI-12 (already standardized, left as is).

Build the inputs with:
    atlas data fldas landsurface --date 1982-01-01:2024-12-31 \
        --coords -71.2,42.2,-70.9,42.5 --out .atlas/data/fldas_1982 --limit 600
    python download_spei.py
"""

from pathlib import Path

import pandas as pd
import torch
from shared_split import HORIZON, SEQ_LEN, spei_cell, split_of
from torch.utils.data import TensorDataset

REPO = Path(__file__).resolve().parents[2]
FLDAS_CSV = REPO / ".atlas/data/fldas_1982/fldas.csv"
SPEI_CSV = REPO / ".atlas/data/spei/spei12.csv"
FEATURES = [
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
]
TARGET = "spei12"


def load_samples(fldas_csv=FLDAS_CSV, spei_csv=SPEI_CSV):
    """All samples before splitting: x (N, SEQ_LEN, F), y (N, 1), and a per-sample frame."""
    # Ocean cells are all-NaN in FLDAS; drop them.
    fldas = pd.read_csv(fldas_csv).dropna(subset=FEATURES)
    spei = pd.read_csv(spei_csv).set_index(["longitude", "latitude", "time"])[TARGET]

    xs, ys, meta = [], [], []
    for (lon, lat), cell in fldas.sort_values("time").groupby(["longitude", "latitude"]):
        months = pd.to_datetime(cell["time"]).dt.to_period("M")
        assert (months.diff().dropna().apply(lambda d: d.n) == 1).all(), (
            f"gap in FLDAS at {lon},{lat}"
        )
        feats = torch.tensor(cell[FEATURES].to_numpy(), dtype=torch.float32)
        s_lon, s_lat = spei_cell(lon, lat)
        for end in range(SEQ_LEN - 1, len(cell)):
            target_month = (months.iloc[end] + HORIZON).strftime("%Y-%m-01")
            key = (round(s_lon, 2), round(s_lat, 2), target_month)
            if key not in spei.index or pd.isna(spei[key]):
                continue
            xs.append(feats[end - SEQ_LEN + 1 : end + 1])
            ys.append(float(spei[key]))
            meta.append((lon, lat, target_month, split_of(target_month)))

    frame = pd.DataFrame(meta, columns=["longitude", "latitude", "target_month", "split"])
    return torch.stack(xs), torch.tensor(ys).unsqueeze(1), frame


def load_splits(fldas_csv=FLDAS_CSV, spei_csv=SPEI_CSV):
    x, y, frame = load_samples(fldas_csv, spei_csv)
    idx = {
        name: torch.tensor(frame.index[frame["split"] == name].to_numpy())
        for name in ("train", "val", "test")
    }
    mean, std = x[idx["train"]].mean((0, 1)), x[idx["train"]].std((0, 1)) + 1e-8
    x = (x - mean) / std
    return {name: TensorDataset(x[i], y[i]) for name, i in idx.items()}
