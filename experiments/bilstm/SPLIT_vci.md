# VCI → FLDAS root-zone soil moisture

A second task on the shared split v2 cells: predict next month's FLDAS root-zone
soil-moisture anomaly from 12 months of the Vegetation Condition Index (VCI). Code:
`download_vci.py`, `data_vci.py`, `train_vci.py`. Results: [`RESULTS_vci.md`](RESULTS_vci.md).
The SPEI-12 task ([`SPLIT_v2.md`](SPLIT_v2.md)) is unchanged.

## Task

| | |
|---|---|
| Cells | The 359 kept 0.5° cells of split v2 (Kansas 88, Texas 209, California 62), same masks |
| Input | VCI at months t-11..t (cell mean) plus month-of-year sin/cos; optionally the soil-moisture anomaly at t-11..t |
| Label | FLDAS 0–100 cm soil-moisture anomaly at t+1 |
| Model | `BiLSTM` from `bilstm.py`, unchanged; hidden 64 |
| Training | `train_v2_delta.py` recipe: AdamW weight decay 1e-2, early stopping (patience 4), batch 256, state shares by effective independent cells, seeds 42–44 |

## VCI (input)

- **Source**: MODIS Terra MOD13C2 v061, monthly NDVI on the 0.05° climate-modeling grid,
  2000-02 → 2024-12 (299 months). `download_vci.py` subsets each month on the server through
  LP DAAC OPeNDAP to W −124.525 … E −93.425, S 25.775 … N 42.025 (326 × 623 pixels,
  ~390 KB per month, 84 MB in all, 4 minutes).
- **Pixels**: kept when pixel reliability is 0 (good) or 1 (marginal); fill, snow/ice and
  cloudy pixels are dropped. NDVI = stored value / 10,000.
- **VCI** per pixel and calendar month: 100 × (NDVI − min) / (max − min), with min and max
  over **2000–2014 only** (the train-input years), so val and test never inform the scaling.
  Pixels whose train-year NDVI range is under 0.05 are dropped as unusable. Values are
  clipped to [−50, 150], because test months can fall outside the train-year range.
- **Cell value**: mean VCI of the valid pixels whose centers fall in the 0.5° cell (median
  100 pixels per cell-month; 93 in California). A cell-month with no valid pixel carries the
  previous month forward (causal). That happened in 0.08% of Kansas and 0.67% of California
  cell-months, and never in Texas.
- **Scaling into the model**: (VCI − 50) / 25.

## Soil moisture (label)

- FLDAS (the same v2 pulls) `SoilMoi00_10cm_tavg`, `SoilMoi10_40cm_tavg`,
  `SoilMoi40_100cm_tavg`, averaged per 0.5° cell as in v2, then weighted by layer
  thickness (0.1, 0.3, 0.6 m) into a 0–100 cm volumetric soil moisture.
- **Anomaly** per cell and calendar month: (value − mean) / std over 1982–2014 (all
  years before validation). The label is that anomaly at t+1, unscaled after that.

## Split

The v1/v2 time blocks by target month, except that training starts later because VCI
starts in 2000-02: the first 12-month window is 2000-02 … 2001-01.

| Split | Target months | Samples |
|---|---|---|
| train | 2001-02 → 2014-12 | 59,953 (KS 14,696 · TX 34,903 · CA 10,354) |
| gap | 2015, 2020 | dropped |
| val | 2016-01 → 2019-12 | 17,232 |
| test | 2021-01 → 2024-12 | 17,232 |

## Variants

| Key | Input | Target |
|---|---|---|
| `vci_only` | VCI + month sin/cos (3 channels) | anomaly at t+1 |
| `vci_sm_change` | VCI + month + soil-moisture anomaly history (4) | change from the anomaly at t |
| `sm_change` | month + soil-moisture anomaly history (3), **no VCI** | change from the anomaly at t |

`sm_change` is an ablation: the same model without VCI, to measure what VCI adds.

Baselines: **climatology** (anomaly 0) and **persistence** (anomaly at t).

## Rebuild

```bash
python experiments/bilstm/download_vci.py     # needs EARTHDATA_TOKEN
python experiments/bilstm/data_vci.py         # needs the v2 FLDAS pulls (see SPLIT_v2.md)
python experiments/bilstm/train_vci.py
```

## Known limits

- MODIS NDVI starts in 2000, so there are 14 train years instead of 32.
- One sensor (Terra). Aqua (MYD13C2, from 2002-07) could be averaged in.
- VCI min/max from 15 years per calendar month is a short reference; long-record VCI
  products (AVHRR, from 1981) use 30+ years.
- The label is a land-surface model output, not measured soil moisture.

## Daily inference (VCI only)

`download_vci16.py`, `data_vci_daily.py`, `train_vci_daily.py`, `predict_vci_daily.py`.
VCI is the **only** input: no soil-moisture history and no month-of-year encoding.

- **NDVI source**: MODIS 16-day NDVI on the same 0.05° grid, Terra MOD13C1 (571 composites
  from 2000-02-18) and Aqua MYD13C1 (518 from 2002-07-04). Together they give a new composite
  about every 8 days. These products are not on OPeNDAP, so each ~105 MB global file is
  cropped to the box and deleted (1,089 files, 19 minutes, 304 MB kept).
- **VCI** per pixel and composite: 100 × (NDVI − min) / (max − min), with min/max over
  2000–2014 for the same sensor and 16-day slot. Same pixel rules and clip as the monthly
  version; cell value = mean over valid pixels.
- **Daily VCI**: on day d, a cell's VCI is that of the latest composite already complete by d
  (a composite starting on s is used from s + 16). Days never see a later composite; a check in
  development confirmed that changing every day after d leaves d's input unchanged.
- **Input for issue day d**: 12 blocks of 30 days (d−359 … d), each the mean daily VCI,
  scaled (VCI − 50) / 25. Shape (12, 1).
- **Label**: mean FLDAS 0–100 cm soil-moisture anomaly over d+1 … d+30. FLDAS is monthly, so
  each day carries its month's anomaly and the 30 days are averaged.
- **Split**: a sample belongs to a split only when its 30-day label window lies wholly inside
  it (train 2001-01-01 … 2014-12-31, val 2016–2019, test 2021–2024); windows that straddle a
  boundary are dropped. First issue day: 2001-04-16.
- **Training**: issue days every 5 days (357,564 train and 102,674 val samples), level target,
  same recipe and seeds as above. **Test**: every issue day of 2020-12-31 … 2024-12-01
  (514,088 samples).
- **Inference**: `python predict_vci_daily.py --date YYYY-MM-DD [--out file.csv]` averages the
  three seed models and writes one row per cell (VCI of the last 30 days, predicted anomaly
  for the next 30 days, and the actual anomaly and persistence when FLDAS covers the window).
  Dates after 2024-12-31 need newer composites, downloaded with `download_vci16.py --end`, and
  a rebuild with `data_vci_daily.py`.
