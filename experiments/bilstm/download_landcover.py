"""Download the MCD12Q1 (MODIS land cover, 500 m, annual) tiles covering a bbox.

Atlas's modis_land_cover client finds the granules; the HDF files are fetched here
with the Earthdata bearer token, because `atlas data modis landcover` is redirected
to the Earthdata login (401) for LP DAAC protected files.

    PYTHONPATH=src uv run python experiments/bilstm/download_landcover.py --year 2010 --coords=-124.48,25.84,-93.51,42.01
"""

import argparse
import asyncio
import os
from datetime import date
from pathlib import Path

import httpx

from atlas.data.base import BBox, PullRequest
from atlas.data.modis_land_cover import MODISLandCoverClient

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / ".atlas/data/mcd12q1"


async def tile_urls(year: int, bbox: BBox) -> list[str]:
    client = MODISLandCoverClient()
    try:
        request = PullRequest(
            start_date=date(year, 1, 1), end_date=date(year, 12, 31), bbox=bbox, limit=100
        )
        result = await client.search(request)
    finally:
        await client.aclose()
    urls = set()
    for scene in result.scenes:
        urls.update(a.href for a in scene.assets.values() if a.href.endswith(".hdf"))
    return sorted(urls)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2010)
    parser.add_argument("--coords", default="-124.48,25.84,-93.51,42.01", help="W,S,E,N")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    west, south, east, north = (float(v) for v in args.coords.split(","))
    urls = asyncio.run(tile_urls(args.year, BBox(west=west, south=south, east=east, north=north)))
    out = args.out / str(args.year)
    out.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {os.environ['EARTHDATA_TOKEN']}"}
    with httpx.Client(follow_redirects=True, timeout=300, headers=headers) as http:
        for url in urls:
            path = out / Path(url).name
            if path.exists():
                continue
            response = http.get(url)
            response.raise_for_status()
            path.write_bytes(response.content)
            print(f"{path.name}: {len(response.content) / 1e6:.1f} MB")
    print(f"{len(urls)} tiles in {out}")


if __name__ == "__main__":
    main()
