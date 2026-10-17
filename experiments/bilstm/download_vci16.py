"""Download MODIS 16-day NDVI (MOD13C1 Terra + MYD13C1 Aqua, v061, 0.05 deg CMG) for daily VCI.

The 16-day products are not on LP DAAC OPeNDAP, so each ~105 MB global HDF file is
streamed to a scratch file, cropped to the Kansas/Texas/California box (the same rows and
columns as download_vci.py) and deleted. Terra and Aqua composites are offset by 8 days,
so together they give a new NDVI composite about every 8 days.

    python download_vci16.py            # 2000-02 .. 2024-12, needs EARTHDATA_TOKEN
"""

import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
from download_vci import COLS, ROWS
from pyhdf.SD import SD, SDC

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / ".atlas/data/mod13c1"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
PRODUCTS = ("MOD13C1", "MYD13C1")
NDVI = "CMG 0.05 Deg 16 days NDVI"
RELIABILITY = "CMG 0.05 Deg 16 days pixel reliability"
WORKERS = 6


def granules(short_name, start, end):
    """(composite start date, https URL) for every granule of ``short_name`` in range."""
    out, page = [], 1
    with httpx.Client(timeout=60) as http:
        while True:
            response = http.get(
                CMR,
                params={
                    "short_name": short_name,
                    "version": "061",
                    "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z",
                    "page_size": 200,
                    "page_num": page,
                },
            )
            response.raise_for_status()
            items = response.json()["items"]
            for item in items:
                umm = item["umm"]
                day = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
                urls = [
                    u["URL"]
                    for u in umm["RelatedUrls"]
                    if u.get("Type") == "GET DATA" and u["URL"].endswith(".hdf")
                ]
                if urls:
                    out.append((day, urls[0]))
            if len(items) < 200:
                return sorted(out)
            page += 1


def fetch(short_name, day, url, out_dir, scratch, headers):
    path = out_dir / f"{short_name}_{day}.npz"
    if path.exists():
        return path.name, "cached", 0
    tmp = scratch / f"{short_name}_{day}.hdf"
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
        r0, r1 = ROWS[0], ROWS[1] + 1
        c0, c1 = COLS[0], COLS[1] + 1
        ndvi = sd.select(NDVI)[r0:r1, c0:c1]
        reliability = sd.select(RELIABILITY)[r0:r1, c0:c1]
        sd.end()
        size = tmp.stat().st_size
    finally:
        tmp.unlink(missing_ok=True)
    np.savez_compressed(path, ndvi=ndvi, reliability=reliability)
    return path.name, "saved", size


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2000-02-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(sn, day, url) for sn in PRODUCTS for day, url in granules(sn, args.start, args.end)]
    print(
        f"{len(jobs)} composites: "
        + ", ".join(f"{sn} {sum(j[0] == sn for j in jobs)}" for sn in PRODUCTS),
        flush=True,
    )
    headers = {"Authorization": f"Bearer {os.environ['EARTHDATA_TOKEN']}"}
    start, total = time.time(), 0
    with (
        tempfile.TemporaryDirectory(prefix="mod13c1_") as scratch,
        ThreadPoolExecutor(WORKERS) as pool,
    ):
        futures = [
            pool.submit(fetch, sn, day, url, args.out, Path(scratch), headers)
            for sn, day, url in jobs
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            name, status, size = future.result()
            total += size
            if done % 50 == 0 or done == len(futures):
                rate = total / 1e6 / (time.time() - start)
                print(f"{done}/{len(futures)} {name} {status} ({rate:.0f} MB/s)", flush=True)


if __name__ == "__main__":
    main()
