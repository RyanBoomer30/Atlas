"""Hidden-size sweep for the change-target model (train_v2_delta.py), test MSE per size.

Everything but ``hidden_size`` is as in train_v2_delta.py: SPEI-history inputs, dSPEI
target, AdamW weight decay 1e-2 (chosen at hidden 64 and kept fixed), early stopping,
effective-sample-size state shares, seeds 42-44. No spatial holdout.

    python sweep_hidden.py
"""

import json

import pandas as pd
import train_v2_delta as D
from data_v2 import load_samples
from train_v2 import equivalence

SIZES = (64, 128, 256)
WEIGHT_DECAY = 1e-2
OUT = D.OUT / "metrics_v2_hidden"


def main():
    OUT.mkdir(exist_ok=True)
    samples = load_samples(spei_history=True)
    n_eff = D.effective_cells(samples)
    shares = {s: n / sum(n_eff.values()) for s, n in n_eff.items()}
    test = samples["split"] == "test"
    rows, runs = [], []
    for hidden in SIZES:
        for seed in D.SEEDS:
            run = D.fit(
                samples, shares, f"hidden{hidden}_seed{seed}", seed, WEIGHT_DECAY, hidden=hidden
            )
            pred = D.forecast(run["model"], samples, test)
            rows += [r | {"hidden": hidden} for r in D.evaluate(pred, samples, test, seed, "test")]
            params = sum(p.numel() for p in run["model"].parameters())
            info = {"hidden": hidden, "seed": seed, "params": params}
            info |= {k: run[k] for k in ("best_epoch", "epochs_run", "best_val", "seconds")}
            if seed == D.SEEDS[0]:
                info["equivalence_pass"] = equivalence(run["model"])[1]
            runs.append(info)
            print(info)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "test_by_seed.csv", index=False)
    pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
    model = table[table.model == "bilstm_delta"]
    persist = table[(table.model == "persistence") & (table.seed == D.SEEDS[0])]
    persist = persist[persist.hidden == SIZES[0]].set_index("group").mse
    summary = (
        model.groupby(["hidden", "group"])
        .mse.agg(["mean", "min", "max"])
        .reset_index()
        .assign(vs_persistence=lambda d: d["mean"] / d.group.map(persist) - 1)
    )
    summary.to_csv(OUT / "test_summary.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "sizes": SIZES,
                "weight_decay": WEIGHT_DECAY,
                "seeds": D.SEEDS,
                "persistence_mse": persist.to_dict(),
                "runs": runs,
            },
            indent=2,
        )
    )
    print(summary.round(4).to_string(index=False))
    assert all(r.get("equivalence_pass", True) for r in runs), "equivalence check FAILED"


if __name__ == "__main__":
    main()
