# Results — shared split v2 (Kansas, Texas, California)

Protocol, masks and normalization: [`SPLIT_v2.md`](SPLIT_v2.md). Every number below comes
from `python train_v2.py` on the data built by `python data_v2.py`; raw tables are in
[`metrics_v2/`](metrics_v2/) and the trained weights are
[`scratch_bilstm_v2.pt`](scratch_bilstm_v2.pt) (state_dict, 40,577 parameters, 171 KB).

## Setup in one line

Scratch `BiLSTM` (unchanged; `input_size=14`), hidden 64, Adam 1e-3, MSE, batch 256,
state-balanced sampling, seed 42, 30 epochs max, checkpoint with best `val_loss`.

| | train | val | test | cells |
|---|---|---|---|---|
| Kansas | 33,792 | 4,224 | 4,224 | 88 |
| Texas | 80,256 | 10,032 | 10,032 | 209 |
| California | 23,808 | 2,976 | 2,976 | 62 |
| **Total** | **137,856** | **17,232** | **17,232** | **359** |

**Best epoch: 1 of 30** (val MSE 0.265). Val loss rises from epoch 2 on while train loss
keeps falling (0.218 → 0.024 by epoch 30), so the kept checkpoint is the first epoch
(≈ 539 optimizer steps). The spatial-holdout model also peaks at epoch 1 (val 0.256).

## Test (2021-01 → 2024-12, all kept cells)

| Group | Model | n | MSE | MAE | Pearson r |
|---|---|---|---|---|---|
| All states | **BiLSTM** | 17,232 | **0.248** | **0.396** | **0.812** |
| | train mean | | 1.161 | 0.886 | — |
| | persistence | | 0.083 | 0.209 | 0.939 |
| Kansas | **BiLSTM** | 4,224 | **0.219** | **0.373** | **0.717** |
| | train mean | | 0.965 | 0.822 | — |
| | persistence | | 0.050 | 0.174 | 0.926 |
| Texas | **BiLSTM** | 10,032 | **0.265** | **0.411** | **0.773** |
| | train mean | | 1.103 | 0.853 | — |
| | persistence | | 0.083 | 0.214 | 0.930 |
| California | **BiLSTM** | 2,976 | **0.232** | **0.379** | **0.914** |
| | train mean | | 1.633 | 1.091 | — |
| | persistence | | 0.128 | 0.241 | 0.954 |

Train mean = 0.155 (one constant, so r is undefined). Persistence = SPEI-12 at t.

## Spatial check (unseen 2°×2° blocks, test years)

8 of 41 blocks (seed 42) holding 78 cells were removed from train and val; a second model
was trained without them and scored on those cells in 2021–2024. "Main model" is the
model above, which did train on these cells (earlier years), on the same samples.

| Group | n | BiLSTM, blocks unseen | Main model (blocks seen) | train mean | persistence |
|---|---|---|---|---|---|
| All | 3,744 | 0.232 (r 0.852) | 0.231 (r 0.855) | 1.196 | 0.085 (r 0.949) |
| Kansas | 480 | 0.142 (r 0.704) | 0.149 (r 0.689) | 0.620 | 0.058 (r 0.894) |
| Texas | 1,872 | 0.268 (r 0.793) | 0.271 (r 0.792) | 1.069 | 0.067 (r 0.948) |
| California | 1,392 | 0.215 (r 0.911) | 0.204 (r 0.918) | 1.566 | 0.120 (r 0.954) |

MSE unless noted; MAE is in `metrics_v2/spatial_holdout_metrics.csv`.

## Equivalence check (trained weights vs `nn.LSTM`)

`check_equivalence.run` on the best checkpoint: **PASS** in float32 (max |diff|
2.4e-7, atol 1e-6) and float64 (max |diff| 2.2e-16, atol 1e-12), for seq 1, 12 and 200,
with one bias and with the bias split across `bias_ih`/`bias_hh`.

## Figures

- [Loss curves, best epoch marked](figures_v2/loss_curves.png)
- [Predicted vs actual, per state](figures_v2/pred_vs_actual.png)
- [Per-state MSE and r vs both baselines](figures_v2/metrics_by_state.png)
- [Same, spatial holdout](figures_v2/spatial_holdout_metrics.png)
- [Map of kept SPEI cells by state, masked cells and holdout blocks](figures_v2/cells_map.png)
- [Equivalence check](figures_v2/equivalence.png)

## Honest read

- **Beats the train-mean baseline everywhere** — overall MSE 0.248 vs 1.161 (−79%), and
  in every state. The FLDAS anomalies carry real drought signal: r 0.81 overall, 0.91 in
  California.
- **Loses clearly to persistence everywhere** — 0.248 vs 0.083 overall (3.0× the
  error); 4.4× in Kansas, 3.2× in Texas, 1.8× in California. On this task (next-month
  SPEI-12, 11 of 12 months shared with SPEI-12 at t) a model that does not see past SPEI
  is not competitive with copying the last value. This is the headline result for the
  FLDAS-only inputs; adding SPEI history fixes most of it (see the variant below).
- **No sign of memorizing places.** On unseen 2°×2° blocks the held-out model matches the
  model that trained on those cells (0.232 vs 0.231), so what it learned transfers across
  space within these three states.
- **It overfits after one epoch.** Train MSE goes 0.218 → 0.024 while val MSE goes
  0.265 → 0.375. With ~138k strongly autocorrelated samples from 359 cells, the
  effective sample size is small for a 40.6k-parameter model with no regularization.
- **Run history.** A first full run had a normalization bug (per-cell std ≈ 0 gave
  |z| up to ~40,000 and constant predictions for two Kansas cells in test); its test MSE
  was 0.279. The std floor and ±10 clip fixed it (see SPLIT_v2.md); only the fixed run's
  numbers are reported above.

## Variant: + SPEI-12 history as input (`train_v2.py --spei-history`)

Same split, cells, masks, seed and hyperparameters; one extra input channel, SPEI-12 at
t-11..t (unscaled), so `input_size=15`. SPEI-12 at t uses climate up to t only, so the
t+1 target does not leak; its last month is exactly the persistence forecast. Outputs:
[`metrics_v2_spei/`](metrics_v2_spei/), [`figures_v2_spei/`](figures_v2_spei/),
[`scratch_bilstm_v2_spei.pt`](scratch_bilstm_v2_spei.pt) (173 KB).

**Best epoch: 2 of 30** (val MSE 0.078); val rises after, train keeps falling
(0.121 → 0.007). Equivalence check: **PASS** (float32 max |diff| 1.5e-7, float64 4.4e-16).

Test (2021–2024), MSE (Pearson r):

| Group | n | BiLSTM + SPEI history | vs persistence | persistence | FLDAS-only v2 | train mean |
|---|---|---|---|---|---|---|
| All states | 17,232 | **0.0774** (0.942) | −6.5% | 0.0828 (0.939) | 0.248 | 1.161 |
| Kansas | 4,224 | 0.0687 (0.908) | **+37%** | 0.0501 (0.926) | 0.219 | 0.965 |
| Texas | 10,032 | 0.0796 (0.933) | −4.3% | 0.0832 (0.930) | 0.265 | 1.103 |
| California | 2,976 | **0.0825** (0.969) | −35% | 0.1275 (0.954) | 0.232 | 1.633 |

Spatial check (same 8 blocks / 78 cells, test years), MSE:

| Group | Blocks unseen | Main model (blocks seen) | persistence |
|---|---|---|---|
| All | 0.0924 (+8.2% vs persistence) | 0.0780 (−8.7%) | 0.0854 |
| Kansas | 0.0581 (tie) | 0.0545 | 0.0581 |
| Texas | 0.1044 (+57%) | 0.0846 | 0.0666 |
| California | 0.0881 (−27%) | 0.0772 | 0.1201 |

Read:

- **SPEI history is the input that mattered.** Test MSE drops 69% (0.248 → 0.077) and the
  model now edges past persistence overall (−6.5%). That is the first teacher on this
  split that beats every baseline in aggregate.
- **The margin is small and uneven.** The overall win comes from California (−35%) and a
  little from Texas (−4%); in Kansas the model is 37% *worse* than persistence. One seed,
  autocorrelated samples: treat the overall −6.5% as promising, not settled.
- **It no longer transfers cleanly to unseen places.** On held-out blocks it loses to
  persistence (+8%), while the model that saw those cells beats it (−9%). Unlike the
  FLDAS-only run, part of the gain is location-specific.
- **Overfitting is still immediate** (best epoch 2), so regularization is the next lever.

## Recommendation for teacher size

Keep the teacher at **hidden 64** (40.6k parameters) and use the SPEI-history inputs.
Both runs peak at epoch 1–2 and only overfit after, so the limit is regularization and
data, not capacity; a larger teacher would overfit faster. Next, in order: (1) add
weight decay or dropout on the head, and repeat over 3+ seeds so the ~6% margin over
persistence gets an error bar; (2) try predicting ΔSPEI-12 = SPEI(t+1) − SPEI(t), which
makes persistence the zero point and may help Kansas; (3) only then sweep 32 / 64 / 128
and pick the smallest size within noise of the best val MSE. Distill only from a teacher
that beats persistence on the spatial holdout as well as on the temporal test.

## Runtimes

| Step | Wall time |
|---|---|
| FLDAS pull, Kansas (5 chunks, incl. 2 retried) | 6.2 min |
| FLDAS pull, Texas (5 chunks, incl. 3 retried) | 9.0 min |
| FLDAS pull, California (5 chunks, incl. 1 retried) | 5.8 min |
| Refetch skipped months (Kansas 5, California 7) | ~3 min |
| SPEI-12 download + three cuts | 46 s |
| MCD12Q1 2010, 11 tiles (~97 MB) | ~1 min |
| `data_v2.py` (aggregate, masks, windows) | 2.3 min |
| Training, main model (30 epochs, CPU, 4 cores) | 10.0 min |
| Training, spatial-holdout model | 8.0 min |
| `data_v2.py` rebuild with SPEI history cached | 5.7 min |
| Training, + SPEI history, main model | 11.2 min |
| Training, + SPEI history, spatial-holdout model | 8.8 min |

With 8 concurrent granule reads, a 108-month Texas chunk (14,124 cells × 12 variables)
took 56–90 s, i.e. under 1 s per month, against the ~5 s per month the sequential extractor took.

## Blockers and caveats from the run

- Earthdata intermittently answered a granule read with a 401 or a login page right
  after a burst of requests; the extractor treats that as an auth failure and stops the
  chunk. Every affected chunk succeeded on retry (6 retries in 15 chunks). Separately,
  12 months were skipped after `peer closed connection without sending complete message
  body (incomplete chunked read)`, which the client does not retry; all 12 were refetched.
- `atlas data modis landcover ... --yes` fails on LP DAAC files: the request is
  redirected to `urs.earthdata.nasa.gov/oauth/authorize` and gets a 401, because the
  download does not send the Earthdata token. `download_landcover.py` works around it.
- All hosts were reachable (CMR, SPEI, Census, LP DAAC); `EARTHDATA_TOKEN` was set.
