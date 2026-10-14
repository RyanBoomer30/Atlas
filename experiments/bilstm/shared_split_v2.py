"""Team shared split v2: three states, one sample stream per 0.5 deg SPEI cell.

The time blocks are v1's, unchanged (imported, not copied): train targets
1983-01..2014-12, val 2016-01..2019-12, test 2021-01..2024-12, gaps 2015 and 2020.
Input is 12 months t-11..t; target is SPEI-12 at t+1. See SPLIT_v2.md.

What v2 changes:
- Region: Kansas, Texas and California (one FLDAS pull per state).
- Sample unit: a 0.5 deg SPEI cell, whose input is the mean of every valid 0.1 deg
  FLDAS cell inside it, so input and label share a footprint.
- Cell masks: state border, hyper-arid / unstable SPEI, cropland or grassland share.
- Features: z-scored per cell and calendar month (train years only) plus month sin/cos.
- States are balanced in training by sampling.
"""

from shared_split import HORIZON, SEED, SEQ_LEN, SPEI_CELL, SPLITS, spei_cell, split_of

__all__ = ["HORIZON", "SEED", "SEQ_LEN", "SPEI_CELL", "SPLITS", "spei_cell", "split_of"]

SPLIT_VERSION = "v2"

# W, S, E, N boxes used for the FLDAS and SPEI pulls, and the Census state FIPS code.
STATES = {
    "kansas": {"bbox": (-102.05, 36.99, -94.59, 40.00), "fips": "20"},
    "texas": {"bbox": (-106.65, 25.84, -93.51, 36.50), "fips": "48"},
    "california": {"bbox": (-124.48, 32.53, -114.13, 42.01), "fips": "06"},
}

FIRST_MONTH, LAST_MONTH = "1982-01", "2024-12"

# Feature statistics (per cell, per calendar month) use input months in these years only.
# The last train target is 2014-12, so no train input is later than 2014-11.
STATS_YEARS = (1982, 2014)

# Hyper-arid rule: drop a cell when more than ARID_MAX_FRACTION of its months in
# STATS_YEARS have mean rainfall below ARID_RAIN_MM mm/month.
ARID_RAIN_MM = 1.0
ARID_MAX_FRACTION = 0.25
# Unstable SPEI rule: drop a cell with any NaN/inf SPEI-12 between FIRST_MONTH and
# LAST_MONTH, or any |SPEI-12| above SPEI_ABS_MAX (a sign the distribution fit failed).
SPEI_ABS_MAX = 4.0

# Land-cover rule: keep a cell when at least LAND_MIN_FRACTION of its MCD12Q1 land
# pixels (IGBP LC_Type1, year LAND_YEAR) are grassland, cropland or cropland mosaic.
LAND_YEAR = 2010
LAND_CLASSES = (10, 12, 14)
LAND_MIN_FRACTION = 0.30

# Spatial check: whole BLOCK_DEG x BLOCK_DEG blocks held out of train and val.
BLOCK_DEG = 2.0
BLOCK_HOLDOUT_FRACTION = 0.2


def block_of(longitude: float, latitude: float) -> tuple[int, int]:
    """Index of the BLOCK_DEG x BLOCK_DEG block containing a point."""
    return int(longitude // BLOCK_DEG), int(latitude // BLOCK_DEG)
