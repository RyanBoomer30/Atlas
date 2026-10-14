"""Download a global SPEIbase file (CSIC, 0.5 deg, monthly) and cut it to a bbox.

Writes a long CSV shaped like fldas.csv (longitude, latitude, time, speiNN) so the
two tables can be joined as features + target.

    python download_spei.py                                   # SPEI-12, FLDAS bbox
    python download_spei.py --scale 3 --coords=-100,35,-95,40 --start 1982-01
    python download_spei.py --coords=-102.05,36.99,-94.59,40.00 --csv kansas_spei12.csv
"""

import argparse
from pathlib import Path

import httpx
import xarray as xr

BASE_URL = "https://spei.csic.es/spei_database_2_11/nc/spei{scale:02d}.nc"
REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / ".atlas/data/spei"
HALF_CELL = 0.25


def download(scale: int, out_dir: Path) -> Path:
    url = BASE_URL.format(scale=scale)
    path = out_dir / Path(url).name
    if path.exists():
        print(f"using cached {path}")
        return path
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".nc.part")
    print(f"downloading {url} (~380 MB)")
    with (
        httpx.stream("GET", url, timeout=60, follow_redirects=True) as response,
        open(partial, "wb") as f,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes(1 << 20):
            f.write(chunk)
    partial.rename(path)
    return path


def subset(path: Path, scale: int, west, south, east, north, start, end, csv_name=None) -> Path:
    ds = xr.open_dataset(path, engine="h5netcdf")
    # Keep every 0.5 deg cell whose footprint overlaps the bbox, so a bbox smaller
    # than one cell still gets the cell it sits in.
    lon, lat = ds["lon"], ds["lat"]
    cells = ds.sel(
        lon=lon[(lon - HALF_CELL < east) & (lon + HALF_CELL > west)],
        lat=lat[(lat - HALF_CELL < north) & (lat + HALF_CELL > south)],
        time=slice(start, end),
    )
    column = f"spei{scale:02d}"
    df = (
        cells["spei"]
        .to_dataframe(name=column)
        .reset_index()
        .rename(columns={"lon": "longitude", "lat": "latitude"})
        .dropna(subset=[column])  # ocean / no-data cells
    )
    df["time"] = df["time"].dt.strftime("%Y-%m-01")
    out = path.with_name(csv_name or f"{column}.csv")
    df[["longitude", "latitude", "time", column]].to_csv(out, index=False)
    n_cells = df.groupby(["longitude", "latitude"]).ngroups
    print(f"wrote {out}: {len(df)} rows, {n_cells} cells, {df['time'].min()} to {df['time'].max()}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=int, default=12, help="SPEI time scale in months (1-48)")
    parser.add_argument("--coords", default="-71.2,42.2,-70.9,42.5", help="W,S,E,N")
    parser.add_argument("--start", default="1982-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default=None, help="last month, YYYY-MM (default: latest)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--csv", default=None, help="CSV name in --out (default: speiNN.csv)")
    args = parser.parse_args()

    west, south, east, north = (float(v) for v in args.coords.split(","))
    path = download(args.scale, args.out)
    subset(path, args.scale, west, south, east, north, args.start, args.end, args.csv)


if __name__ == "__main__":
    main()
