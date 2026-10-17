# Results — VCI → FLDAS root-zone soil moisture

Setup: [`SPLIT_vci.md`](SPLIT_vci.md). Every number comes from `python train_vci.py`;
raw tables in [`metrics_vci/`](metrics_vci/). Seeds 42–44; mean (min–max) over seeds.
Training is deterministic: a full re-run reproduced every test MSE exactly.

## Test (2021–2024), MSE of the soil-moisture anomaly

| Model | All states | Kansas | Texas | California | r (all) |
|---|---|---|---|---|---|
| Climatology (anomaly 0) | 0.934 | 0.792 | 0.890 | 1.281 | — |
| Persistence (anomaly at t) | 0.313 | 0.240 | 0.352 | 0.287 | 0.786 |
| **VCI only** | 0.783 (0.769–0.796) | 0.736 | 0.731 | 1.023 | 0.418 |
| SM history only, change target | 0.2926 (0.2867–0.2978) | 0.2304 | **0.3239** | **0.2756** | 0.783 |
| **VCI + SM history, change target** | **0.2921** (0.2867–0.2949) | **0.2242** | 0.3254 | 0.2759 | 0.783 |

Relative to persistence:

| Model | All | Kansas | Texas | California |
|---|---|---|---|---|
| VCI only | +150% | +206% | +108% | +257% |
| SM history only | −6.6% | −4.1% | **−8.0%** | −3.9% |
| VCI + SM history | **−6.8%** | **−6.7%** | −7.5% | −3.8% |

95% interval of MSE(model) − MSE(persistence), resampling whole test months:
VCI + SM history −0.0399…+0.0005, −0.0386…−0.0003, −0.0469…−0.0109 (seeds 42, 43, 44);
SM history only −0.0409…−0.0006, −0.0353…+0.0035, −0.0471…−0.0095.

## Spatial check (8 unseen 2°×2° blocks, 78 cells, test years)

| Model | All | Kansas | Texas | California |
|---|---|---|---|---|
| Persistence | 0.301 | 0.130 | 0.359 | 0.282 |
| VCI only | 0.927 (+208%) | 0.914 (+606%) | 0.772 | 1.141 |
| SM history only | **0.278 (−7.8%)** | 0.130 (+0.6%) | **0.321 (−10.6%)** | 0.270 (−4.4%) |
| VCI + SM history | 0.279 (−7.4%) | **0.123 (−5.2%)** | 0.326 (−9.1%) | **0.269 (−4.8%)** |

## Figures

- [MSE vs persistence, with and without VCI](figures_vci/skill_vs_persistence.png)
- [State-mean test series: actual, persistence and each model](figures_vci/test_series.png)

## Read

- **VCI does carry soil-moisture signal on its own**, but much less than the soil-moisture
  record itself. VCI-only cuts error 16% against climatology (0.783 vs 0.934, r 0.42), yet
  it is 2.5× persistence's error. Vegetation lags and smooths soil moisture, and the
  test-series figure shows VCI-only tracking the slow swings and missing the month-to-month
  moves. Training also peaks later (best epoch 3–6, not 1), a sign of a harder, weaker
  signal.
- **With soil-moisture history, the change-target model beats persistence by about 7%**,
  the same recipe that worked for SPEI. It wins in every state and on unseen blocks. The
  bootstrap interval excludes zero for 2 of 3 seeds and just touches it for the third.
- **VCI adds almost nothing on top of soil-moisture history overall**: 0.2921 with VCI vs
  0.2926 without, inside the seed spread (per seed −1.0% to +0.4%). The exception is
  **Kansas**: VCI lowers MSE in all three seeds (−1.9% to −3.6%) and on unseen Kansas
  blocks (−5.2% vs +0.6% without). Kansas is cropland and grassland, where vegetation
  responds quickly to root-zone moisture. In Texas, VCI slightly hurts (+0.6% and +1.4% in
  2 of 3 seeds).
- **Bottom line**: if the operational question is "estimate soil moisture where we have
  satellite vegetation data but no soil-moisture record", VCI alone gives a weak but real
  estimate (r 0.42). If the soil-moisture history is available, VCI is only worth adding
  for Kansas-like cropland.

## Daily inference, VCI only (`train_vci_daily.py`)

A forecast issued every day of 2021–2024 for the next 30 days (514,088 samples). Input is
VCI alone. Mean over seeds 42–44 (min–max); "ensemble" averages the three models as
`predict_vci_daily.py` does.

| Test MSE | All states | Kansas | Texas | California | r (all) |
|---|---|---|---|---|---|
| Climatology (anomaly 0) | 0.883 | 0.757 | 0.829 | 1.240 | — |
| Persistence (last 30 days) | 0.218 | 0.172 | 0.243 | 0.199 | — |
| **VCI only, daily** | **0.749** (0.735–0.757) | 0.671 | 0.710 | 0.992 | 0.353 |
| VCI only, daily, 3-seed ensemble | 0.745 | 0.666 | 0.706 | 0.986 | 0.355 |
| vs climatology | −15.1% | −11.4% | −14.3% | −20.0% | |

- **Issue days on month-ends only**: 0.802, −14.2% vs climatology (0.935). The monthly
  MOD13C2 VCI-only model above scored 0.783 on the same kind of target, so issuing daily from
  16-day composites costs nothing in skill and gives a fresh forecast every day.
- **Unseen 2°×2° blocks**: 0.869 vs climatology 0.927 (−6.3%). Texas −13.9% and California
  −3.9% hold up, but **Kansas is worse than climatology on unseen blocks** (+23.4%,
  r 0.08). VCI's link to soil moisture in Kansas does not transfer to Kansas cells the model
  never saw.
- **Persistence is still far better** (0.218): the VCI-only model has 3.4× its error, and the
  bootstrap interval of the gap is +0.43 … +0.63 for every seed. Persistence needs the
  soil-moisture record, which this model deliberately does not use.
- Best epoch 1 for every seed (spatial seed 42: epoch 2); about 3.8 minutes per model.
- Equivalence check on seed 42: PASS.

![Daily forecasts, 2021-2024](figures_vci_daily/test_daily_series.png)

The figure shows what VCI can and cannot do. The model follows the slow drought swings
(Kansas 2022–23, Texas 2022) but lags or misses sharp wet-ups (California winter 2022–23),
because vegetation greens weeks after the soil wets. It also stays near the middle where
the actual anomaly is extreme.

## Equivalence and runtimes

`check_equivalence.run` on seed 42 of every variant: PASS (float32 and float64).
MOD13C2 download 4.0 min; `data_vci.py` 3.1 min; training all 18 models (3 variants ×
3 seeds × main + spatial) about 10 minutes on CPU.

## Next steps

- Nowcast instead of forecast (VCI at t → soil moisture at t). VCI responds to past moisture,
  so the t+1 horizon may undersell it.
- Add Aqua MYD13C2 and EVI or VCI at shorter lags (16-day MOD13A2) for a sharper signal.
- Try the 10–40 cm layer as the label, which vegetation roots sample most directly.
