"""Download MODIS 8-day land-surface temperature (MOD11C2 / MYD11C2 v061, 0.05 deg CMG) for TCI.

Like download_vci16.py: each ~59 MB global HDF file is streamed to a scratch file, cropped to
the contiguous-US box (shared_split_us.CMG_ROWS / CMG_COLS, the same pixels as the US NDVI
crop) and deleted. Keeps daytime and nighttime LST (uint16, x 0.02 K, 0 = fill) and their QC
bytes as lst / qc and lst_night / qc_night. Terra (MOD11C2) passes at about 10:30 and 22:30
local solar time, Aqua (MYD11C2, from 2002-07) at about 13:30 and 01:30.

    python download_lst8.py                      # Terra, 2000-02 .. 2024-12, needs EARTHDATA_TOKEN
    python download_lst8.py --product MYD11C2    # Aqua
"""

import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
from download_vci16 import WORKERS, granules
from pyhdf.SD import SD, SDC
from shared_split_us import CMG_COLS, CMG_ROWS, MOD11C2_US_DIR, MYD11C2_US_DIR

OUT_DIRS = {"MOD11C2": MOD11C2_US_DIR, "MYD11C2": MYD11C2_US_DIR}


def complete(path):
    """True when the file exists and already holds the nighttime fields."""
    if not path.exists():
        return False
    with np.load(path) as f:
        return "lst_night" in f.files


def fetch(product, day, url, out_dir, scratch, headers):
    path = out_dir / f"{product}_{day}.npz"
    if complete(path):
        return path.name, "cached", 0
    tmp = scratch / f"{product}_{day}.hdf"
    for attempt in range(4):
        try:
            with (
                httpx.Client(follow_redirects=True, timeout=600, headers=headers) as http,
                http.stream("GET", url) as response,
            ):
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_bytes(1 << 20):
                        f.write(chunk)
            break
        except (httpx.HTTPError, OSError):
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    try:
        sd = SD(str(tmp), SDC.READ)
        rows = slice(CMG_ROWS[0], CMG_ROWS[1] + 1)
        cols = slice(CMG_COLS[0], CMG_COLS[1] + 1)
        fields = {
            key: sd.select(name)[rows, cols]
            for key, name in (
                ("lst", "LST_Day_CMG"),
                ("qc", "QC_Day"),
                ("lst_night", "LST_Night_CMG"),
                ("qc_night", "QC_Night"),
            )
        }
        sd.end()
        size = tmp.stat().st_size
    finally:
        tmp.unlink(missing_ok=True)
    np.savez_compressed(path, **fields)
    return path.name, "saved", size


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2000-02-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--product", choices=list(OUT_DIRS), default="MOD11C2")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.out = args.out or OUT_DIRS[args.product]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = granules(args.product, args.start, args.end)
    print(f"{len(jobs)} composites", flush=True)
    headers = {"Authorization": f"Bearer {os.environ['EARTHDATA_TOKEN']}"}
    start, total = time.time(), 0
    with (
        tempfile.TemporaryDirectory(prefix="mod11c2_") as scratch,
        ThreadPoolExecutor(WORKERS) as pool,
    ):
        futures = [
            pool.submit(fetch, args.product, day, url, args.out, Path(scratch), headers)
            for day, url in jobs
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            name, status, size = future.result()
            total += size
            if done % 50 == 0 or done == len(futures):
                rate = total / 1e6 / (time.time() - start)
                print(f"{done}/{len(futures)} {name} {status} ({rate:.0f} MB/s)", flush=True)


if __name__ == "__main__":
    main()
