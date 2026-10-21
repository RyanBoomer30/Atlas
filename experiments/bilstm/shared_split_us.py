"""Contiguous-US extension of the VCI-only setup (see SPLIT_vci.md, "Plains and contiguous US").

Same 0.5 deg cells, masks, daily VCI input and FLDAS labels as the three-state setup, but
pulled once for the whole contiguous US box and split into states by the Census borders.
Alaska, Hawaii and Puerto Rico are left out.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / ".atlas/data"
FLDAS_US_DIR = DATA / "fldas_us"
MOD13C1_US_DIR = DATA / "mod13c1_us"
OUT_US_DIR = DATA / "vci_us"

# W, S, E, N on 0.5 deg cell edges, so no cell is cut by the box.
BOX = (-125.0, 24.5, -66.5, 49.5)
# FLDAS pull tiles: longitude edges x latitude edges (whole degrees, never splitting a cell).
TILE_LON = (-125.0, -119.0, -113.0, -107.0, -101.0, -95.0, -89.0, -83.0, -77.0, -71.0, -66.5)
TILE_LAT = (24.5, 37.0, 49.5)
DATE_CHUNKS = (
    ("1982-01-01", "1990-12-31"),
    ("1991-01-01", "1999-12-31"),
    ("2000-01-01", "2008-12-31"),
    ("2009-01-01", "2017-12-31"),
    ("2018-01-01", "2024-12-31"),
)
# Only what the VCI setup uses: the three soil layers (labels) and rain (hyper-arid mask).
FLDAS_US_VARIABLES = [
    "SoilMoi00_10cm_tavg",
    "SoilMoi10_40cm_tavg",
    "SoilMoi40_100cm_tavg",
    "Rainf_f_tavg",
]
# Rows/cols of the 3600 x 7200 MODIS CMG 0.05 deg grid covering BOX plus one pixel on each
# side (pixel centers 49.525 .. 24.425 N, -125.025 .. -66.475 E).
CMG_ROWS, CMG_COLS = (809, 1311), (1099, 2270)

EXCLUDED_FIPS = ("02", "15", "72")  # Alaska, Hawaii, Puerto Rico
THREE_STATES = ("kansas", "texas", "california")
# Great Plains states: the two already used plus the four between them and Canada.
PLAINS = ("north_dakota", "south_dakota", "nebraska", "kansas", "oklahoma", "texas")
