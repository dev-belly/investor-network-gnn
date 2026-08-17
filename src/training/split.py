"""Time-series aware train / val / test splits.

We perform a single chronological split (no k-fold to keep the run-time in
check) but enforce a *purge gap* between train / val and val / test to avoid
information leakage from labels with overlapping forward windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TimeSplit:
    train: tuple[pd.Timestamp, pd.Timestamp]
    val: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]

    def as_dict(self) -> dict[str, tuple[str, str]]:
        return {
            "train": [self.train[0].isoformat(), self.train[1].isoformat()],
            "val": [self.val[0].isoformat(), self.val[1].isoformat()],
            "test": [self.test[0].isoformat(), self.test[1].isoformat()],
        }


def chronological_split(
    dates: pd.DatetimeIndex,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
    test_ratio: float = 0.25,
    purge_gap: int = 5,
) -> TimeSplit:
    """Split a sorted date index into train/val/test with a purge gap.

    The gap is enforced *only* at the boundaries (val starts ``purge_gap`` days
    after train ends, test starts ``purge_gap`` days after val ends).  This
    protects the model from seeing labels whose forward window overlaps the
    training period.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    n = len(dates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_train < 10 or n_val < 5 or n_test < 5:
        raise ValueError(f"Too few dates for a meaningful split: n={n}")

    train_end = dates[n_train - 1]
    val_start = dates[n_train + purge_gap]
    val_end = dates[n_train + n_val - 1]
    test_start = dates[n_train + n_val + purge_gap]
    test_end = dates[-1]
    return TimeSplit(
        train=(dates[0], train_end),
        val=(val_start, val_end),
        test=(test_start, test_end),
    )


def filter_by_split(
    panel: pd.DataFrame, split: TimeSplit, split_name: str
) -> pd.DataFrame:
    start, end = getattr(split, split_name)
    mask = (panel.index.get_level_values("date") >= start) & (
        panel.index.get_level_values("date") <= end
    )
    return panel.loc[mask]


def select_graph_dates(
    dates: pd.DatetimeIndex, freq: int = 20, *, end: Optional[pd.Timestamp] = None
) -> pd.DatetimeIndex:
    """Pick a sparse set of dates for graph construction.

    Building a graph every day is wasteful; the network changes slowly.  We
    re-build it every ``freq`` trading days.
    """
    if end is None:
        end = dates[-1]
    idx = dates[dates <= end]
    if len(idx) == 0:
        return idx
    # Always keep the last day
    last = idx[-1]
    sampled = idx[::freq]
    if sampled[-1] != last:
        sampled = sampled.append(pd.DatetimeIndex([last]))
    return pd.DatetimeIndex(sorted(sampled.unique()))
