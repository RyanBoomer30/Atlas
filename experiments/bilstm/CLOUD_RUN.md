# Cloud run: drought split v2 (Kansas, Texas, California)

Paste everything below the line into a Claude Code session at claude.ai/code started on
**RyanBoomer30/Atlas**, branch **bilstm-shared-split**, in an environment that has:

- env var `EARTHDATA_TOKEN` (copy the value from your local `.env`)
- network access to `*.earthdata.nasa.gov`, `*.earthdatacloud.nasa.gov`, `spei.csic.es`,
  `www2.census.gov`, and the Python package registry

Or just tell the session: "Follow experiments/bilstm/CLOUD_RUN.md".

---

You are running unattended. Nobody will answer questions: make reasonable decisions, record
them, and finish with pushed branches plus a results write-up.

## Setup

- This session's `origin` is the fork `RyanBoomer30/Atlas`. Work from `origin/bilstm-shared-split`.
  Never push to `AerospaceNU/Atlas` and never push to `main`.
- First, confirm the environment and stop early if it is wrong:
  - `test -n "$EARTHDATA_TOKEN" && echo token-set`. Never print, log, or commit the token's
    value, and never commit `.env`.
  - `curl -sI https://cmr.earthdata.nasa.gov/search/ | head -1`,
    `curl -sI https://spei.csic.es/spei_database_2_11/nc/spei12.nc | head -1`,
    `curl -sI https://www2.census.gov/ | head -1`. Record which hosts are reachable.
  - If the token is missing or both Earthdata and SPEI are unreachable, write the exact errors
    to `experiments/bilstm/RESULTS_v2.md`, push it, report, and stop.
- Python: `uv sync` for the Atlas package. If `import atlas` fails, run the CLI as
  `PYTHONPATH=src uv run atlas ...`. Install the experiment dependencies in a separate venv:
  `uv venv /tmp/exp && VIRTUAL_ENV=/tmp/exp uv pip install -r experiments/bilstm/requirements.txt`.
  CPU torch is fine.
- Read first: `experiments/bilstm/SPLIT.md`, `shared_split.py`, `data.py`, `bilstm.py`,
  `train.py`, `check_equivalence.py`, `download_spei.py`, and `src/atlas/data/sources/fldas.py`.

## Task 1: faster FLDAS extraction (branch `fldas-concurrent-extract`, from `origin/main`)

`FldasClient.extract_training_table` in `src/atlas/data/sources/fldas.py` awaits one OPeNDAP
request per monthly granule, one after another (~5 s each, so 516 months ≈ 45 min).

- Run granule reads concurrently with a bounded limit (`asyncio.Semaphore`, default 8).
- Keep output rows in the same deterministic order as today.
- Keep the existing retry on 502/503/504 and sensible progress output.
- Add or adjust tests in `tests/test_fldas.py`; all existing tests must pass.
- Run CI's checks: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`,
  `uv run pytest`.
- Commit (message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`) and push
  to `origin` as `fldas-concurrent-extract`. Then merge it into the v2 branch below so the pulls
  use it.

## Task 2: shared split v2 (branch `bilstm-split-v2`, from `origin/bilstm-shared-split`)

Keep the v1 time blocks exactly: train targets 1983-01..2014-12, val 2016-01..2019-12, test
2021-01..2024-12, gaps 2015 and 2020; input = 12 months t-11..t, target = SPEI-12 at t+1.
Don't edit v1 in `shared_split.py`; add v2 alongside it (for example `shared_split_v2.py`) so
v1 stays reproducible.

1. **Pulls, one per state.** The client requests the whole bbox span every month, so one box
   from California to Kansas would be huge. Use dates `1982-01-01:2024-12-31`,
   `--limit 600 --yes`, and the default 10 variables plus `Rainf_f_tavg,Tair_f_tavg`.
   Approximate bboxes (W,S,E,N):
   - Kansas `-102.05,36.99,-94.59,40.00`
   - Texas `-106.65,25.84,-93.51,36.50`
   - California `-124.48,32.53,-114.13,42.01`

   Write to `.atlas/data/fldas_v2/<state>/`. Cut SPEI-12 to the same boxes with
   `download_spei.py`, extending it if needed.
2. **Sampling.** One sample stream per 0.5° SPEI cell. The input is the mean of all valid 0.1°
   FLDAS cells inside that SPEI cell, matching the label's footprint, not a single point.
3. **Cell masks.** Record each rule and how many cells it removed.
   - Clip to state borders (SPEI cell center inside the state) using US Census cartographic
     boundaries, e.g. https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip.
     If blocked, fall back to the bbox and say so.
   - Drop hyper-arid or unstable-SPEI cells: e.g. more than 25% of train-period months with
     near-zero `Rainf_f_tavg`, or NaN/inf runs in SPEI-12. Choose a defensible threshold and
     document it.
   - Keep cells with at least 30% cropland or grassland using Atlas `modis_land_cover`
     (MCD12Q1) if reachable. If the network or file format blocks it within reasonable
     effort, skip it and document that it is outstanding.
4. **Features.** Z-score each feature per cell and per calendar month using train years only,
   so inputs are anomalies. Add month of year as sin/cos. Leave the SPEI-12 target unscaled.
5. **Balance states.** Weight the loss (or sample) so each state contributes equally during
   training, and report per-state counts.
6. **Training.** The `BiLSTM` class in `bilstm.py` keeps its exact method names and
   signatures; only `input_size` changes, through the constructor. Use
   `L.seed_everything(42)`, hidden_size 64, Adam lr 1e-3, MSE, at most 30 epochs, and keep the
   checkpoint with the best `val_loss` (record its epoch). Batch size 256, documented as a v2
   change from 16. CPU is fine.
7. **Evaluation.** Report test MSE, MAE and Pearson r, overall and per state, against two
   baselines: predict the train mean, and persistence (SPEI-12 at t predicting t+1). Add a
   spatial check: hold out whole 2°×2° blocks (about 20% of blocks, seed 42) in the test
   years and report the same metrics. Re-run `check_equivalence.py` on the trained weights;
   it must PASS.
8. **Figures** in `experiments/bilstm/figures_v2/`, in the same style as `train.py` (light
   surface; Kansas blue `#2a78d6`, Texas orange `#eb6834`, California aqua `#1baf7a`; legends
   plus direct labels; no dual axes):
   - loss curves with the best epoch marked
   - predicted vs actual, one panel per state
   - per-state metric bars against both baselines
   - map scatter of the kept SPEI cells colored by state
   - equivalence-check bars
9. **Docs.** Write `experiments/bilstm/SPLIT_v2.md` (task, exact pull commands, masks with
   counts, normalization, balancing, protocol, known limits) and
   `experiments/bilstm/RESULTS_v2.md` (all metric tables, figure links, an honest read of
   whether the model beats the baselines, and a recommendation for teacher size).
10. **Commit and push.**
    - Run `uv run ruff check experiments` and `uv run ruff format --check experiments`.
    - Commit code, docs, figures, metrics (CSV/JSON), and the best checkpoint if it fits the
      500 KB pre-commit limit. Otherwise leave the checkpoint out and say so.
    - Never commit raw data (`.atlas/`) or logs.
    - Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
    - Push to `origin` as `bilstm-split-v2`.

## If something blocks you

For network denials, Earthdata 401/403, or unreachable hosts, don't retry in a loop. Write
the exact error into `RESULTS_v2.md`, finish everything that doesn't depend on it, push what
you have, and report. Never make up metrics: every number in the docs must come from a run
you executed.

## Final report

- branches pushed, with commit SHAs
- overall and per-state test metrics against both baselines, and the best epoch
- equivalence-check result
- which masks were applied or skipped, and why
- runtimes of the pulls and of training
- any blockers, with the exact error text
