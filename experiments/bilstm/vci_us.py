"""VCI models on more states: three states, six Plains states, West / East of 98 W, or the US.

Setup as vci_tactics.py ("uniform"): daily VCI is the only input, 10-40 cm soil-moisture
label, raw output, rolling origin 2021-2024 (train 2001 .. year before), one epoch over every
training sample, seeds 42-44 (ensemble = seed mean). States are drawn in proportion to their
effective number of independent cells. All three regions use the contiguous-US build
(data_us.py), so they differ only in which states they train on.

--inputs tci adds daily TCI (data_us.py --tci) as a second channel, built exactly like the VCI
channel (12 x 30-day block means, same scaling).

--inputs coords appends standardized latitude and longitude (vci_coords.add_coordinates,
statistics over the region's cells) to the VCI input.

--spatial trains without held-out 2x2 deg blocks and scores only those: the 8 blocks the
three-state runs held out (chosen by train_v2.spatial_holdout over the three-state cells),
plus 20 % of the other states' blocks (same rule and seed).

    python vci_us.py --region plains
    python vci_us.py --region us --years 2021 --spatial
    python vci_us.py --region us --inputs coords
    python vci_us.py --region us --time-only      # time 3 % of one epoch, print the estimate
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import train_v2_delta as D
from data_us import load
from data_vci_daily import first_anchor, samples_for
from data_vci_daily import load_daily as load_three_state
from shared_split_us import EAST, OUT_US_DIR, PLAINS, THREE_STATES, WEST
from shared_split_v2 import BLOCK_HOLDOUT_FRACTION, SEED, block_of
from train_v2 import equivalence
from train_v2 import spatial_holdout as holdout_blocks
from train_vci_daily import TRAIN_STRIDE
from train_vci_daily_recent import decompose, window_period
from vci_coords import add_coordinates
from vci_tactics import SEEDS, TEST_YEARS, fit_one_epoch, predict, select

OUT = D.OUT
METRICS = OUT / "metrics_vci_us"
LABEL = "sm_10_40"
REGIONS = {"three": THREE_STATES, "plains": PLAINS, "west": WEST, "east": EAST, "us": None}


def cells_of(daily, states):
    keep = np.isin(daily["names"][daily["state"]], states)
    cell_keys = ("vci", "tci", "sm", "state", "lon", "lat")
    return {k: (v[keep] if k in cell_keys else v) for k, v in daily.items()}


def effective_cells(samples, names):
    """Per state: effective number of independent cells from train-period label changes."""
    frame = pd.DataFrame(
        {
            "state": samples["state"],
            "cell": samples["lon"].astype(str) + "," + samples["lat"].astype(str),
            "anchor": samples["target_month"],
            "dy": samples["y"] - samples["persistence"],
        }
    )
    out = {}
    for i in np.unique(samples["state"]):
        wide = frame[frame.state == i].pivot(index="anchor", columns="cell", values="dy")
        if wide.shape[1] == 1:
            out[names[i]] = 1.0
            continue
        eig = np.linalg.eigvalsh(np.corrcoef(wide.to_numpy().T))
        out[names[i]] = float(eig.sum() ** 2 / (eig**2).sum())
    return out


def held_cells(daily):
    """(lon, lat) of every cell in a held-out block.

    Blocks: the 8 the three-state runs held out (whole blocks, so cells of a neighbouring
    state inside them are held out too), plus 20 % of the blocks with no three-state cell.
    """
    three = load_three_state()
    held, _, _ = holdout_blocks({"lon": three["lon"], "lat": three["lat"]})
    three_blocks = {block_of(a, b) for a, b in zip(three["lon"], three["lat"], strict=True)}
    chosen = {block_of(a, b) for a, b in zip(three["lon"][held], three["lat"][held], strict=True)}
    blocks = [block_of(a, b) for a, b in zip(daily["lon"], daily["lat"], strict=True)]
    other = sorted(set(blocks) - three_blocks)
    rng = np.random.default_rng(SEED)
    picks = rng.permutation(len(other))[: round(BLOCK_HOLDOUT_FRACTION * len(other))]
    chosen |= {other[i] for i in picks}
    return {
        (round(float(a), 2), round(float(b), 2))
        for a, b, block in zip(daily["lon"], daily["lat"], blocks, strict=True)
        if block in chosen
    }


def in_cells(samples, cells):
    pairs = zip(samples["lon"].round(2).tolist(), samples["lat"].round(2).tolist(), strict=True)
    return np.array([p in cells for p in pairs])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", choices=list(REGIONS), required=True)
    parser.add_argument("--inputs", choices=["vci", "coords", "tci"], default="vci")
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--years", type=int, nargs="+", default=list(TEST_YEARS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    METRICS.mkdir(exist_ok=True)
    D.INPUTS = ["vci"]
    daily = load(LABEL)
    names = daily["names"]
    held = held_cells(daily) if args.spatial else set()
    region = daily if REGIONS[args.region] is None else cells_of(daily, REGIONS[args.region])
    days = pd.DatetimeIndex(region["days"])
    first = first_anchor(region)
    if args.inputs == "tci":
        first = max(first, first_anchor(region | {"vci": region["tci"]}))
    anchors = np.arange(first, len(days))
    test_anchors = anchors[days[anchors] >= pd.Timestamp(f"{min(args.years) - 1}-12-01")]
    coarse = samples_for(region, anchors[::TRAIN_STRIDE])
    fine = samples_for(region, test_anchors)
    if args.inputs == "tci":
        as_vci = region | {"vci": region["tci"]}
        for samples, idx in ((coarse, anchors[::TRAIN_STRIDE]), (fine, test_anchors)):
            tci_x = samples_for(as_vci, idx)["x"]
            samples["x"] = np.concatenate([samples["x"], tci_x], axis=2)
            del tci_x

    if args.inputs == "coords":
        lat_stats = (float(region["lat"].mean()), float(region["lat"].std()))
        lon_stats = (float(region["lon"].mean()), float(region["lon"].std()))
        coarse = add_coordinates(coarse, lat_stats, lon_stats)
        fine = add_coordinates(fine, lat_stats, lon_stats)

    base = window_period(coarse, {"t": ("2001-01-01", "2020-12-31")}) == "t"
    n_eff = effective_cells(select(coarse, base), names)
    shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}
    held_coarse = in_cells(coarse, held) if held else np.zeros(len(coarse["y"]), bool)
    held_fine = in_cells(fine, held) if held else np.ones(len(fine["y"]), bool)
    print(
        f"{args.region}: {len(region['state'])} cells in {len(n_eff)} states, "
        f"held-out cells {len(held)}",
        flush=True,
    )

    run = "spatial_unseen" if args.spatial else "test"
    tag = f"{args.region}_{'' if args.inputs == 'vci' else args.inputs + '_'}"
    tag += "spatial" if args.spatial else "main"
    if args.years != list(TEST_YEARS):
        tag += "_" + "-".join(map(str, args.years))
    rows, equivalence_ok = [], None
    for year in args.years:
        train = window_period(coarse, {"t": ("2001-01-01", f"{year - 1}-12-31")}) == "t"
        train &= ~held_coarse
        test = window_period(fine, {"t": (f"{year}-01-01", f"{year}-12-31")}) == "t"
        test &= held_fine
        tr, te = select(coarse, train), select(fine, test)
        counts = np.bincount(tr["state"], minlength=len(names))
        share = np.array([shares.get(s, 0.0) for s in names])
        weights = share[tr["state"]] / counts[tr["state"]]
        if args.time_only:
            draws = max(len(weights) // 33, 20000)
            t0 = time.time()
            fit_one_epoch(tr, weights, SEEDS[0], f"us_timing_{args.region}_{args.inputs}", draws)
            secs = (time.time() - t0) * len(weights) / draws
            print(
                f"TIMING {args.region} {args.inputs} {year}: {len(weights)} training samples, "
                f"{secs / 60:.1f} min per one-epoch model on {args.threads} thread(s)",
                flush=True,
            )
            return
        models = [fit_one_epoch(tr, weights, s, f"us_{tag}_{year}_s{s}") for s in SEEDS]
        if equivalence_ok is None:
            equivalence_ok = equivalence(models[0])[1]
        pred = predict(models, te["x"])
        for i in np.unique(te["state"]):
            sel = te["state"] == i
            y = te["y"][sel].astype(float)
            rows.append(
                {
                    "region": args.region,
                    "inputs": args.inputs,
                    "run": run,
                    "year": year,
                    "state": names[i],
                    "n": int(sel.sum()),
                    "n_train": int(train.sum()),
                    "climatology_mse": float(np.mean(y**2)),
                    "persistence_mse": float(np.mean((te["persistence"][sel] - y) ** 2)),
                }
                | decompose(pred[sel].astype(float), y)
            )
        this_year = [r for r in rows if r["year"] == year]
        sse = sum(r["mse"] * r["n"] for r in this_year)
        clim = sum(r["climatology_mse"] * r["n"] for r in this_year)
        print(tag, year, f"overall skill {1 - sse / clim:+.3f}", flush=True)
        pd.DataFrame(rows).assign(equivalence_pass=equivalence_ok).to_csv(
            METRICS / f"{tag}.csv", index=False
        )
    pd.Series(n_eff).rename("effective_cells").to_csv(METRICS / f"{tag}_effective_cells.csv")


if __name__ == "__main__":
    OUT_US_DIR.mkdir(parents=True, exist_ok=True)
    main()
