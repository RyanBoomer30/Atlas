# Shared split v1 — drought Bi-LSTM comparison

Everyone trains and reports on this split so results line up. Code: `shared_split.py`
(the source of truth) and `data.py` (loader). Don't change v1; propose v2 instead.

## Task

| | |
|---|---|
| Input | 12 consecutive months (t-11..t) of the 10 default FLDAS variables for one 0.1° cell |
| Target | SPEI-12 at month t+1 (SPEIbase v2.11) for the 0.5° SPEI cell containing that FLDAS cell |
| Loss / metric | MSE on SPEI-12 (no target scaling; SPEI is already standardized) |
| Feature scaling | z-score per variable, mean/std from **train only** |
| Seed | 42 (`L.seed_everything(42)`), hidden_size 64, 30 epochs, Adam 1e-3, batch 16 |

## Split (by target month)

| Split | Target months | Years |
|---|---|---|
| train | 1983-01 → 2014-12 | 32 |
| *gap* | 2015 | dropped |
| val | 2016-01 → 2019-12 | 4 |
| *gap* | 2020 | dropped |
| test | 2021-01 → 2024-12 | 4 |

Every cell appears in every split; the holdout is in time.

## Why this split

- **Blocked in time, not random.** SPEI-12 at month t aggregates months t-11..t, so
  consecutive targets share 11/12 of their climate. A random split leaks near-copies of
  test labels into train and overstates skill.
- **12-month gaps.** With a 1-year gap, the months behind every train label are
  disjoint from those behind every val/test label.
- **Most recent years held out.** Matches deployment (predict forward in time) and puts
  recent events, including the 2022 Northeast drought, in test.
- **Train starts 1983-01** because the first full 12-month FLDAS input window is 1982.

## Known limits

- SPEIbase fits its distribution over 1901–2024, so the standardization has seen the
  test years. Unavoidable with the published product.
- Current pull is the Boston bbox only: 8 FLDAS cells but only **2 distinct SPEI-12
  series**. Fine for checking the pipeline, too small to rank models. The split is
  defined by dates, so it carries over unchanged when we pull a larger region.

## Rebuild the data

```bash
atlas data fldas landsurface --date 1982-01-01:2024-12-31 \
  --coords -71.2,42.2,-70.9,42.5 --out .atlas/data/fldas_1982 --limit 600 --yes
python experiments/bilstm/download_spei.py
```
