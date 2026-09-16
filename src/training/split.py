"""Time-series aware train / val / test splits.

We perform a single chronological split (no k-fold to keep the run-time in
check) but enforce a *purge gap* between train / val and val / test to avoid
information leakage from labels with overlapping forward windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
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
    purge_gap: int = 20,
    label_horizon: int = 20,
) -> TimeSplit:
    """Split a sorted date index into train/val/test with a purge gap.

    ``purge_gap`` counts omitted dates from the supplied trading-date index,
    not calendar days. It must cover ``label_horizon`` future trading dates,
    so the previous split's final label ends before the next split begins.
    Ratios describe the pre-purge partitions; validation and test lose their
    first ``purge_gap`` dates. Short post-purge partitions fail explicitly.
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if not all(isfinite(ratio) and 0 < ratio < 1 for ratio in ratios):
        raise ValueError("Split ratios must be finite and strictly between 0 and 1")
    if abs(sum(ratios) - 1.0) >= 1e-6:
        raise ValueError("Split ratios must sum to 1")
    if (
        isinstance(label_horizon, bool)
        or not isinstance(label_horizon, Integral)
        or label_horizon < 1
    ):
        raise ValueError("label_horizon must be a positive integer number of trading dates")
    if (
        isinstance(purge_gap, bool)
        or not isinstance(purge_gap, Integral)
        or purge_gap < label_horizon
    ):
        raise ValueError("purge_gap must be an integer at least as large as label_horizon")
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans:
        raise ValueError("Split dates must not contain NaT")
    dates = dates.unique().sort_values()
    n = len(dates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    actual_val = n_val - purge_gap
    actual_test = n_test - purge_gap
    if n_train < 10 or actual_val < 5 or actual_test < 5:
        raise ValueError(
            "Too few dates after purging: "
            f"train={n_train}, val={actual_val}, test={actual_test}; "
            "need at least 10 train, 5 validation and 5 test dates"
        )

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
