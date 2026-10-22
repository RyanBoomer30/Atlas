"""Record every input, gate, label and output of one US model, for the map report.

Model: VCI + TCI + season + data age (vci_us.py --inputs tci --time), trained on every 30-day
window in 2001-01-01 .. 2023-12-31 of all 2,014 US cells, one epoch, seed 42 (the first seed of
the vci_us.py ensemble). Forecasts are issued on the 1st of each month of 2024.

Per issue day and cell it keeps, for each of the 12 input blocks (oldest first): VCI, TCI, data
ages, and the mean over the 64 hidden units of the forget, input, limbo (candidate) and output
gates, forward and backward; plus the label (10-40 cm anomaly over the next 30 days), the model
output and persistence (anomaly over the last 30 days).

    python map_gates.py                       # writes .atlas/data/vci_us/map_gates.npz
    python map_gates.py --region three --draws 3000 --out /tmp/x.npz   # quick test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import train_v2_delta as D
from bilstm import DIRECTIONS, GATES
from data_us import AVAILABLE_AFTER, LST_AVAILABLE_AFTER, OUT_US_DIR, capture_age, load
from data_vci_daily import LEAD_DAYS, first_anchor, samples_for
from train_v2 import equivalence
from train_vci_daily import TRAIN_STRIDE
from train_vci_daily_recent import window_period
from vci_tactics import fit_one_epoch, select
from vci_us import REGIONS, cells_of, effective_cells, time_channels

SEED = 42
TEST_YEAR = 2024


def build_inputs(region, idx, ages):
    """(samples, 12, 6): VCI, TCI, sin/cos day of year, VCI age / 30, TCI age / 30."""
    samples = samples_for(region, idx)
    tci_x = samples_for(region | {"vci": region["tci"]}, idx)["x"]
    samples["x"] = np.concatenate([samples["x"], tci_x, time_channels(region, idx, ages)], axis=2)
    return samples


@torch.no_grad()
def trace(model, x):
    """Output and per-step mean gate values -> (n,), (n, 2 directions, 12 steps, 4 gates)."""
    n, steps, _ = x.shape
    hidden = model.hparams.hidden_size
    gates = np.zeros((n, len(DIRECTIONS), steps, len(GATES)), np.float32)
    finals = {}
    for d, direction in enumerate(DIRECTIONS):
        long, short = x.new_zeros(n, hidden), x.new_zeros(n, hidden)
        order = range(steps) if direction == "fwd" else reversed(range(steps))
        for t in order:
            step = x[:, t]
            values = {
                "forget": torch.sigmoid(model._gate("forget", step, short, direction)),
                "input": torch.sigmoid(model._gate("input", step, short, direction)),
                "limbo": torch.tanh(model._gate("limbo", step, short, direction)),
                "output": torch.sigmoid(model._gate("output", step, short, direction)),
            }
            long = values["forget"] * long + values["input"] * values["limbo"]
            short = values["output"] * torch.tanh(long)
            for g, gate in enumerate(GATES):
                gates[:, d, t, g] = values[gate].mean(dim=1).numpy()
        finals[direction] = short
    out = model.head(torch.cat([finals["fwd"], finals["bwd"]], dim=1))[:, 0]
    assert torch.allclose(out, model(x)[:, 0], atol=1e-5), "gate trace does not match forward()"
    return out.numpy(), gates


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", choices=list(REGIONS), default="us")
    parser.add_argument(
        "--draws", type=int, default=None, help="training draws (default: one epoch)"
    )
    parser.add_argument("--out", type=Path, default=OUT_US_DIR / "map_gates.npz")
    args = parser.parse_args()
    torch.set_num_threads(1)
    D.INPUTS = ["vci"]
    daily = load("sm_10_40")
    names = daily["names"]
    region = daily if REGIONS[args.region] is None else cells_of(daily, REGIONS[args.region])
    days = pd.DatetimeIndex(region["days"])
    first = max(first_anchor(region), first_anchor(region | {"vci": region["tci"]}))
    anchors = np.arange(first, len(days))
    ages = [
        capture_age(region["vci"], AVAILABLE_AFTER),
        capture_age(region["tci"], LST_AVAILABLE_AFTER),
    ]

    coarse = build_inputs(region, anchors[::TRAIN_STRIDE], ages)
    train = window_period(coarse, {"t": ("2001-01-01", f"{TEST_YEAR - 1}-12-31")}) == "t"
    base = window_period(coarse, {"t": ("2001-01-01", "2020-12-31")}) == "t"
    n_eff = effective_cells(select(coarse, base), names)
    shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}
    tr = select(coarse, train)
    del coarse
    counts = np.bincount(tr["state"], minlength=len(names))
    share = np.array([shares.get(s, 0.0) for s in names])
    weights = share[tr["state"]] / counts[tr["state"]]
    model = fit_one_epoch(tr, weights, SEED, f"map_gates_{args.region}", args.draws)
    del tr
    _, equivalence_ok = equivalence(model)
    torch.save(model.state_dict(), args.out.with_suffix(".pt"))

    issue = pd.date_range(f"{TEST_YEAR}-01-01", f"{TEST_YEAR}-12-01", freq="MS")
    idx = days.get_indexer(issue)
    assert (idx + LEAD_DAYS < len(days)).all()
    frames = build_inputs(region, idx, ages)
    output, gates = trace(model, torch.from_numpy(frames["x"]))
    n, k = len(region["state"]), len(idx)
    raw = frames["x"].reshape(k, n, 12, -1)
    np.savez_compressed(
        args.out,
        issue=issue.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        lon=region["lon"],
        lat=region["lat"],
        state=names[region["state"]],
        vci=raw[..., 0] * 25 + 50,  # back to the 0-100 index scale
        tci=raw[..., 1] * 25 + 50,
        doy_sin=raw[:, 0, :, 2],
        doy_cos=raw[:, 0, :, 3],
        vci_age=raw[..., 4] * 30,
        tci_age=raw[..., 5] * 30,
        gates=gates.reshape(k, n, len(DIRECTIONS), 12, len(GATES)),
        label=frames["y"].reshape(k, n),
        output=output.reshape(k, n),
        persistence=frames["persistence"].reshape(k, n),
        equivalence_pass=equivalence_ok,
    )
    skill = 1 - np.mean((output - frames["y"]) ** 2) / np.mean(frames["y"] ** 2)
    print(f"wrote {args.out}: {k} issue days x {n} cells; skill vs climatology {skill:+.3f}")


if __name__ == "__main__":
    main()
