"""Label-boundary regressions requiring only pandas, NumPy and pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import make_labels
from src.training.split import chronological_split, filter_by_split


@pytest.mark.parametrize("horizon", [5, 20, 40])
def test_actual_forward_label_windows_end_before_next_split(horizon):
    calendar = pd.bdate_range("2020-01-01", periods=700)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (len(calendar), 3)),
        index=calendar,
        columns=["A", "B", "C"],
    )
    labels = make_labels(
        returns, {"label": {"horizon": horizon, "risk_quantiles": [0.2, 0.8]}}
    )
    dates = labels.index.get_level_values("date").unique()
    split = chronological_split(dates, purge_gap=horizon, label_horizon=horizon)

    for previous, following in [("train", "val"), ("val", "test")]:
        previous_rows = filter_by_split(labels, split, previous)
        following_rows = filter_by_split(labels, split, following)
        last_date = previous_rows.index.get_level_values("date").max()
        next_date = following_rows.index.get_level_values("date").min()
        last_position = calendar.get_loc(last_date)
        window = returns.iloc[last_position + 1:last_position + horizon + 1]

        # Check the actual production label against its contributing returns.
        np.testing.assert_allclose(
            labels.loc[last_date, "fwd_ret"].to_numpy(),
            (1 + window).prod().to_numpy() - 1,
        )
        assert len(window) == horizon
        assert window.index[-1] < next_date
        omitted = dates[(dates > last_date) & (dates < next_date)]
        assert len(omitted) == horizon


def test_rejects_old_five_day_gap_for_twenty_day_labels():
    dates = pd.bdate_range("2020-01-01", periods=400)
    with pytest.raises(ValueError, match="at least as large as label_horizon"):
        chronological_split(dates, purge_gap=5, label_horizon=20)


@pytest.mark.parametrize("n_dates,purge_gap", [(0, 20), (100, 20), (160, 20), (300, 80)])
def test_rejects_short_post_purge_partitions(n_dates, purge_gap):
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    with pytest.raises(ValueError, match="Too few dates after purging"):
        chronological_split(dates, purge_gap=purge_gap)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_ratio": float("nan")},
        {"train_ratio": 0.7},
        {"test_ratio": 0},
        {"label_horizon": 0},
        {"label_horizon": 1.5},
        {"label_horizon": True},
        {"purge_gap": 20.5},
        {"purge_gap": True},
    ],
)
def test_rejects_invalid_split_configuration(kwargs):
    with pytest.raises(ValueError):
        chronological_split(pd.bdate_range("2020-01-01", periods=400), **kwargs)


def test_rejects_missing_dates():
    dates = pd.bdate_range("2020-01-01", periods=400).append(pd.DatetimeIndex([pd.NaT]))
    with pytest.raises(ValueError, match="NaT"):
        chronological_split(dates)


def test_sorts_and_deduplicates_dates_before_counting_gap():
    dates = pd.bdate_range("2020-01-01", periods=400)
    assert chronological_split(dates[::-1].append(dates)) == chronological_split(dates)
