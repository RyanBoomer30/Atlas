"""Team shared split v1 for the drought Bi-LSTM comparison. Import this; don't copy it.

Sample: 12 months of FLDAS features (months t-11..t) for one 0.1 deg cell -> SPEI-12
at month t+1 for the 0.5 deg SPEI cell containing it. A sample's split is decided
by its *target month* only, so every member gets identical sets.

Split is blocked in time, not random. Neighbouring SPEI-12 values share 11 of
their 12 aggregated months, so a random split puts near-duplicates of test labels
in train. A 12-month gap between blocks makes the months behind every train label
disjoint from those behind every val/test label.
"""

SPLIT_VERSION = "v1"
SEED = 42
SEQ_LEN = 12  # input months per sample
HORIZON = 1  # target is SPEI-12 this many months after the last input month

# Inclusive target-month ranges (YYYY-MM). Targets in the gaps (2015, 2020) are dropped.
SPLITS = {
    "train": ("1983-01", "2014-12"),
    "val": ("2016-01", "2019-12"),
    "test": ("2021-01", "2024-12"),
}

SPEI_CELL = 0.5


def split_of(target_month: str) -> str | None:
    """Split name for a target month ('YYYY-MM' or 'YYYY-MM-DD'), or None if in a gap."""
    month = target_month[:7]
    for name, (start, end) in SPLITS.items():
        if start <= month <= end:
            return name
    return None


def spei_cell(longitude: float, latitude: float) -> tuple[float, float]:
    """Center of the 0.5 deg SPEIbase cell containing a point."""

    def snap(v):
        return (v // SPEI_CELL) * SPEI_CELL + SPEI_CELL / 2

    return snap(longitude), snap(latitude)
