"""The supervised data helper must keep future labels out of model inputs."""

import pandas as pd
import pytest

from src.data.loader import LoadedData


@pytest.mark.parametrize("bounded", [False, True])
def test_feature_matrix_separates_targets_and_preserves_filtered_row_alignment(bounded):
    dates = pd.bdate_range("2024-01-02", periods=3)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "stock_id"])
    panel = pd.DataFrame(
        {
            "momentum_20d": range(6),
            "roe": range(10, 16),
            "y_dir": [0, 1] * 3,
            "y_risk": [0, 1, 2] * 2,
            "fwd_ret": [0.1, -0.1] * 3,
        },
        index=index,
    )
    feature_names = ["roe", "momentum_20d"]
    label_names = ["y_dir", "y_risk", "fwd_ret"]
    data = LoadedData(None, panel[feature_names], panel[label_names], panel, feature_names)
    bounds = {"start": dates[1], "end": dates[1]} if bounded else None

    X, y = data.feature_matrix(bounds)

    expected = panel.loc[[dates[1]]] if bounded else panel
    pd.testing.assert_frame_equal(X, expected[feature_names])
    pd.testing.assert_frame_equal(y, expected[label_names])
    assert X.columns.intersection(y.columns).empty
