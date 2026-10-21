"""Pull FLDAS for the contiguous-US box (shared_split_us.py) with the Atlas CLI.

The box is cut into whole-degree tiles and the 1982-2024 range into five date chunks; each
(tile, chunk) is one `atlas data fldas landsurface` call, run JOBS at a time and retried.
A call that exits 0 leaves a `.done` marker, so a rerun only redoes what is missing. Months
a chunk still skipped are refetched one month per call into `fill/`.

    python download_fldas_us.py      # uses the repo's .venv atlas CLI
"""

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import pairwise

import pandas as pd
from shared_split_us import (
    DATE_CHUNKS,
    FLDAS_US_DIR,
    FLDAS_US_VARIABLES,
    REPO,
    TILE_LAT,
    TILE_LON,
)

JOBS = 8
TRIES = 4
ENV = os.environ | {"PYTHONPATH": str(REPO / "src")}


def tiles():
    for w, e in pairwise(TILE_LON):
        for s, n in pairwise(TILE_LAT):
            yield f"{w:g}_{s:g}", (w, s, e, n)


def pull(out, start, end, box):
    if (out / ".done").exists():
        return out, "cached", 0
    out.mkdir(parents=True, exist_ok=True)
    command = [
        str(REPO / ".venv/bin/atlas"), "data", "fldas", "landsurface",
        "--date", f"{start}:{end}",
        "--coords", ",".join(f"{v:g}" for v in box),
        "--variables", ",".join(FLDAS_US_VARIABLES),
        "--out", str(out), "--limit", "600", "--yes",
    ]  # fmt: skip
    t0 = time.time()
    for attempt in range(TRIES):
        with open(out.with_suffix(".log"), "w") as log:
            rc = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=ENV
            ).returncode
        if rc == 0:
            (out / ".done").touch()
            return out, f"ok try={attempt + 1}", time.time() - t0
        time.sleep(20 * (attempt + 1))
    return out, f"FAILED rc={rc}", time.time() - t0


def missing_months(tile_dir):
    have = set()
    for path in tile_dir.glob("**/fldas.csv"):
        have |= set(pd.read_csv(path, usecols=["time"])["time"].str[:7])
    want = {str(m) for m in pd.period_range(DATE_CHUNKS[0][0][:7], DATE_CHUNKS[-1][1][:7], "M")}
    return sorted(want - have)


def run(jobs):
    with ThreadPoolExecutor(JOBS) as pool:
        futures = [pool.submit(pull, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            out, status, secs = future.result()
            print(
                f"{done}/{len(jobs)} {out.relative_to(FLDAS_US_DIR)} {status} {secs:.0f}s",
                flush=True,
            )


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    t0 = time.time()
    run(
        [
            (FLDAS_US_DIR / name / "chunks" / start, start, end, box)
            for name, box in tiles()
            for start, end in DATE_CHUNKS
        ]
    )
    fills = []
    for name, box in tiles():
        for month in missing_months(FLDAS_US_DIR / name):
            last = (pd.Period(month, "M").end_time).strftime("%Y-%m-%d")
            fills.append((FLDAS_US_DIR / name / "fill" / month, f"{month}-01", last, box))
    print(f"{len(fills)} months to refetch", flush=True)
    run(fills)
    left = {name: missing_months(FLDAS_US_DIR / name) for name, _ in tiles()}
    print("still missing:", {k: v for k, v in left.items() if v} or "none")
    print(f"ALL_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
