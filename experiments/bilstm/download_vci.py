"""Download MODIS monthly NDVI (MOD13C2 v061, 0.05 deg CMG) for the v2 states, for VCI.

Each month is subset on the server through LP DAAC OPeNDAP to the box around Kansas,
Texas and California (~390 KB per month instead of a ~95 MB global file). Writes one
``.npz`` per month (NDVI as stored, int16 x 1e4, and pixel reliability) plus the grid.

    python download_vci.py            # 2000-02 .. 2024-12, needs EARTHDATA_TOKEN
"""

import argparse
import asyncio
import io
import os
from pathlib import Path
from urllib.parse import quote

import h5py
import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / ".atlas/data/mod13c2"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
SHORT_NAME, VERSION = "MOD13C2", "061"
GROUP = "/MOD_Grid_monthly_CMG_VI"
# Rows/cols of the 3600 x 7200 CMG grid covering W -124.525 .. E -93.425, N 42.025 .. S 25.775
# (cell centers), a little wider than the union of the three state boxes.
ROWS, COLS = (959, 1284), (1109, 1731)
CONCURRENCY = 8


def constraint():
    r, c = f"[{ROWS[0]}:1:{ROWS[1]}]", f"[{COLS[0]}:1:{COLS[1]}]"
    return ";".join(
        [
            f"{GROUP}/Data_Fields/CMG_0_05_Deg_Monthly_NDVI{r}{c}",
            f"{GROUP}/Data_Fields/CMG_0_05_Deg_Monthly_pixel_reliability{r}{c}",
            f"{GROUP}/Latitude{r}",
            f"{GROUP}/Longitude{c}",
        ]
    )


def granules(start, end):
    """(YYYY-MM, OPeNDAP base URL) for every monthly granule in [start, end]."""
    out, page = {}, 1
    with httpx.Client(timeout=60) as http:
        while True:
            response = http.get(
                CMR,
                params={
                    "short_name": SHORT_NAME,
                    "version": VERSION,
                    "temporal": f"{start}-01T00:00:00Z,{end}-28T23:59:59Z",
                    "page_size": 200,
                    "page_num": page,
                },
            )
            response.raise_for_status()
            items = response.json()["items"]
            for item in items:
                umm = item["umm"]
                month = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:7]
                urls = [u["URL"] for u in umm["RelatedUrls"] if "opendap" in u["URL"]]
                if urls:
                    out[month] = urls[0]
            if len(items) < 200:
                return dict(sorted(out.items()))
            page += 1


async def fetch(http, month, base, out_dir, limit):
    path = out_dir / f"{month}.npz"
    if path.exists():
        return month, "cached"
    url = f"{base}.dap.nc4?dap4.ce={quote(constraint(), safe='')}"
    async with limit:
        for attempt in range(4):
            try:
                response = await http.get(url)
                if response.status_code in (502, 503, 504):
                    raise httpx.HTTPStatusError(
                        "retry", request=response.request, response=response
                    )
                response.raise_for_status()
                break
            except (httpx.HTTPError, OSError):
                if attempt == 3:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))
    with h5py.File(io.BytesIO(response.content)) as f:
        group = f[GROUP.strip("/")]
        ndvi = group["Data_Fields/CMG_0_05_Deg_Monthly_NDVI"][:]
        reliability = group["Data_Fields/CMG_0_05_Deg_Monthly_pixel_reliability"][:]
        lat, lon = group["Latitude"][:], group["Longitude"][:]
    grid = out_dir / "grid.npz"
    if not grid.exists():
        np.savez(grid, lat=lat, lon=lon)
    np.savez_compressed(path, ndvi=ndvi, reliability=reliability)
    return month, "saved"


async def main_async(start, end, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    months = granules(start, end)
    print(f"{len(months)} MOD13C2 granules {start} .. {end}")
    headers = {"Authorization": f"Bearer {os.environ['EARTHDATA_TOKEN']}"}
    limit = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=300, follow_redirects=True, headers=headers) as http:
        tasks = [fetch(http, m, base, out_dir, limit) for m, base in months.items()]
        for done, result in enumerate(asyncio.as_completed(tasks), start=1):
            month, status = await result
            if done % 25 == 0 or done == len(tasks):
                print(f"{done}/{len(tasks)} (last {month}: {status})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2000-02")
    parser.add_argument("--end", default="2024-12")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    asyncio.run(main_async(args.start, args.end, args.out))


if __name__ == "__main__":
    main()
