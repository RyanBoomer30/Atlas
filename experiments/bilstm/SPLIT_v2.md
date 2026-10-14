# Shared split v2 — drought Bi-LSTM, Kansas + Texas + California

v2 keeps v1's time blocks and task and changes the region, the sample unit, the cell
masks, the features and the balancing. Code: `shared_split_v2.py` (constants, source of
truth), `data_v2.py` (build + loader), `train_v2.py` (training and evaluation). v1
(`shared_split.py`, `data.py`, `train.py`) is untouched and still reproduces.
Results: [`RESULTS_v2.md`](RESULTS_v2.md).

## Task

| | |
|---|---|
| Sample unit | one 0.5° SPEIbase cell (one sample stream per cell) |
| Input | 12 consecutive months (t-11..t) of 14 features: the 10 default FLDAS variables + `Rainf_f_tavg` + `Tair_f_tavg`, each the **mean of every valid 0.1° FLDAS cell inside the SPEI cell**, plus month-of-year sin/cos |
| Target | SPEI-12 at month t+1 (SPEIbase v2.11) for that cell, **unscaled** |
| Loss / metric | MSE on SPEI-12; also MAE and Pearson r |
| Model | `BiLSTM` from `bilstm.py`, unchanged; only `input_size` (14) changes, through the constructor |
| Training | `L.seed_everything(42)`, hidden 64, Adam 1e-3, MSE, at most 30 epochs, checkpoint with best `val_loss` kept |
| Batch | **256** (v1: 16). v2 has ~138k train samples vs ~3k in v1; 16 would mean ~8.6k steps per epoch on CPU |

## Split (by target month) — identical to v1

| Split | Target months | Years |
|---|---|---|
| train | 1983-01 → 2014-12 | 32 |
| *gap* | 2015 | dropped |
| val | 2016-01 → 2019-12 | 4 |
| *gap* | 2020 | dropped |
| test | 2021-01 → 2024-12 | 4 |

`shared_split_v2.py` imports `SPLITS`, `SEQ_LEN`, `HORIZON`, `SEED` and `split_of` from
`shared_split.py`; it does not copy them.

## Rebuild the data

FLDAS, one pull per state. Each state is one bbox; the date range is cut into five
chunks (1982–1990, 1991–1999, 2000–2008, 2009–2017, 2018–2024) so a chunk that fails can be
re-run alone and the extractor's in-memory table stays small (Texas is 14,124 0.1° cells
× 516 months ≈ 7.3M rows). `extract_training_table` now reads 8 granules concurrently
(branch `fldas-concurrent-extract`).

```bash
V=SoilMoi00_10cm_tavg,SoilMoi10_40cm_tavg,SoilMoi40_100cm_tavg,SoilMoi100_200cm_tavg,Evap_tavg,Qs_tavg,Qsb_tavg,SWE_inst,SnowDepth_inst,RadT_tavg,Rainf_f_tavg,Tair_f_tavg
# per state and chunk (START:END in the five chunks above):
atlas data fldas landsurface --date START:END --coords -102.05,36.99,-94.59,40.00 \
  --variables $V --out .atlas/data/fldas_v2/kansas/chunks/START --limit 600 --yes
atlas data fldas landsurface --date START:END --coords -106.65,25.84,-93.51,36.50 \
  --variables $V --out .atlas/data/fldas_v2/texas/chunks/START --limit 600 --yes
atlas data fldas landsurface --date START:END --coords -124.48,32.53,-114.13,42.01 \
  --variables $V --out .atlas/data/fldas_v2/california/chunks/START --limit 600 --yes
```

The extractor skips a month whose request still fails after its retries (dropped
connections are not retried). Any month missing from a state's chunks was fetched again
on its own into `.atlas/data/fldas_v2/<state>/fill/YYYY-MM/` with the same command and
`--date YYYY-MM-01:YYYY-MM-<last day>`: Kansas 5 months, California 7, Texas none.
`data_v2.py` reads every `fldas.csv` under the state's folder and drops duplicate rows.

SPEI-12, cut to the same boxes (`download_spei.py` gained `--csv`):

```bash
python experiments/bilstm/download_spei.py --coords=-102.05,36.99,-94.59,40.00 --csv kansas_spei12.csv
python experiments/bilstm/download_spei.py --coords=-106.65,25.84,-93.51,36.50 --csv texas_spei12.csv
python experiments/bilstm/download_spei.py --coords=-124.48,32.53,-114.13,42.01 --csv california_spei12.csv
```

State borders and land cover:

```bash
mkdir -p .atlas/data/census && cd .atlas/data/census
curl -O https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip && unzip cb_2023_us_state_20m.zip
cd - && PYTHONPATH=src uv run python experiments/bilstm/download_landcover.py --year 2010
```

`download_landcover.py` finds the MCD12Q1 granules with Atlas's `modis_land_cover`
client and fetches the HDF tiles with the Earthdata bearer token (11 tiles, ~97 MB).
`atlas data modis landcover ... --yes` itself fails: LP DAAC redirects it to the
Earthdata login and it gets a 401, so the download does not send the token there.

Then `python experiments/bilstm/data_v2.py` (experiment venv; needs `pyshp` and `pyhdf`)
writes `.atlas/data/fldas_v2/samples_v2.npz`, `masks_v2.json` and `cells_v2.csv`, and
`python experiments/bilstm/train_v2.py` trains and evaluates.

## Sampling

FLDAS rows with any missing feature (ocean / no-data) are dropped, then the remaining
0.1° cells are averaged per SPEI cell and month. A 0.1° cell belongs to the 0.5° cell
containing its center (`shared_split.spei_cell`). Kept cells average 10–25 FLDAS cells
(median 25, i.e. full coverage); partial cells sit on coasts and box edges.

## Cell masks

Applied in this order; each count is cells removed by that rule from those still kept.

| Rule | Kansas | Texas | California |
|---|---|---|---|
| SPEI cells with ≥ 1 valid FLDAS cell (start) | 96 | 549 | 312 |
| SPEI-12 present (non-ocean in SPEIbase) | −0 | −0 | −0 |
| Cell center inside the state border (Census `cb_2023_us_state_20m`) | −8 | −285 | −139 |
| FLDAS input present in every month 1982-01..2024-12 | −0 | −0 | −0 |
| Hyper-arid: > 25% of 1982–2014 months with < 1 mm rain (`Rainf_f_tavg`) | −0 | −0 | −27 |
| Unstable SPEI-12: any NaN/inf, any \|SPEI\| > 4, or a run of ≥ 6 identical months | −0 | −0 | −0 |
| MCD12Q1 2010 IGBP grassland (10), cropland (12) or cropland/natural mosaic (14) < 30% of land pixels | −0 | −55 | −84 |
| **Kept** | **88** | **209** | **62** |

Thresholds. *Hyper-arid*: SPEI's log-logistic fit is unreliable where many months have
no rain; 1 mm/month is "effectively dry", and 25% (3 dry months a year on average) is the
threshold from the task brief. It removes the Mojave/Colorado desert cells and keeps the
Central Valley (largest kept dry-month share: California 0.247, Texas 0.010, Kansas 0.003, so West Texas stays).
*Unstable SPEI*: SPEIbase v2.11 had no NaN, inf or \|SPEI\| > 3.5 and no repeated
consecutive value anywhere in the three boxes, so this rule removes nothing; it stays as
a guard. *Land cover*: drought impact on range and crops is the use case; forest,
shrubland and barren cells are dropped. One year (2010, mid-train) is used for all
months; land-cover change over 1982–2024 is ignored.

## Normalization

Each of the 12 FLDAS features is z-scored **per cell and per calendar month** with the
mean and std of that cell's months in 1982–2014 (the train years; the last train input
month is 2014-11). Inputs are therefore anomalies from the cell's own seasonal cycle.
Some cell/month pairs barely vary in the train years (baseflow `Qsb_tavg` in dry cells,
snow in the south), so a plain z-score turns a later change into |z| up to ~40,000; that
saturated the LSTM on a first run (constant predictions for whole cells in val/test).
Each per-cell std is therefore floored at 0.1 × the feature's pooled 1982–2014 std over
the state's kept cells, and z is clipped to ±10 (0.016% of input values reach the clip). Month of year is added as sin and cos of 2π(month−1)/12. The SPEI-12 target
is left unscaled.

## Balancing

Texas has 3.4× California's cells. Training draws each batch with a
`WeightedRandomSampler` (weight 1 / that state's train-sample count, with replacement,
seeded), so every state contributes a third of each epoch in expectation; an epoch is
still the train-set size. `BiLSTM.training_step` is unchanged (plain MSE). Validation and
test are unweighted; results are reported per state.

| Samples | train | val | test | cells |
|---|---|---|---|---|
| Kansas | 33,792 | 4,224 | 4,224 | 88 |
| Texas | 80,256 | 10,032 | 10,032 | 209 |
| California | 23,808 | 2,976 | 2,976 | 62 |

## Protocol

- Test metrics overall and per state: MSE, MAE, Pearson r.
- Baselines: **train mean** (mean train-split SPEI-12, one constant) and **persistence**
  (SPEI-12 at t, the last input month, predicts t+1).
- Spatial check: the region is cut into 2°×2° blocks (`floor(lon/2), floor(lat/2)`); 20%
  of the blocks that hold a kept cell are drawn with `numpy.random.default_rng(42)`. A
  second model is trained with those blocks removed from train and val, then scored on
  the held-out blocks in the test years, with both baselines (train mean recomputed
  without the held-out blocks). The main model's score on the same cells is reported
  next to it.
- `check_equivalence.run` on the trained weights, float32 and float64; must PASS.

## Known limits

- SPEIbase fits its distribution over 1901–2024, so the standardization has seen the
  test years (as in v1).
- The model sees FLDAS anomalies only, not past SPEI. SPEI-12 at t+1 shares 11 of its 12
  aggregated months with SPEI-12 at t, so persistence is a very strong baseline that this
  input set cannot use directly.
- Neighbouring cells are strongly correlated, so the effective sample size is far below
  the sample count, and the temporal test still contains every training cell. The
  spatial check covers unseen places only for ~20% of blocks and one seed.
- Cells on box edges and coasts average fewer than 25 FLDAS cells; the state border test
  uses the SPEI cell center only.
- One land-cover year (2010); the land mask uses MCD12Q1 pixels whose center falls in
  the 0.5° cell.
