"""Small real-model checks for checkpoint reuse and evaluation isolation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gcn import GCNClassifier, predict_gcn
from src.training.metrics import compute_auc, compute_ece, top_decile_hit_rate
from src.training.trainer import _snapshots_to_arrays, gcn_edge_drop_curve, train_gcn
from src.utils.common import save_json, set_seed


def panel_fixture():
    dates = pd.bdate_range("2020-01-01", periods=6)
    stocks = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([dates, stocks], names=["date", "stock_id"])
    panel = pd.DataFrame({"x": np.arange(24) / 24, "y_dir": [0, 1] * 12}, index=index)
    edge = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    snapshots = {date: SimpleNamespace(edge_index=edge, edge_weight=np.ones(6)) for date in dates}
    return panel, snapshots, stocks


def test_zero_drop_reuses_exact_checkpoint_and_predictions(monkeypatch):
    panel, snapshots, stocks = panel_fixture()
    set_seed(7)
    model = GCNClassifier(in_dim=1, hidden_dims=[4], dropout=0.4)
    X, y, ei, ew, _ = _snapshots_to_arrays(panel, snapshots, stocks, ["x"])
    main_probs = predict_gcn(model, X, ei, ew)
    checkpoint = {key: value.clone() for key, value in model.state_dict().items()}

    def reject_training(*args, **kwargs):
        raise AssertionError("An inference ablation must not train a model")

    monkeypatch.setattr("src.training.trainer.train_gcn", reject_training)
    cfg = {"data": {"seed": 7}, "ablation": {"edge_drop_rates": [0.0, 0.5]}}
    curve = gcn_edge_drop_curve(model, panel, snapshots, stocks, ["x"], cfg)
    reverse = gcn_edge_drop_curve(model, panel, snapshots, stocks, ["x"], cfg, drop_rates=[0.5, 0.0])
    assert curve[0.0] == compute_auc(y.numpy(), main_probs)
    assert curve == reverse
    np.testing.assert_array_equal(main_probs, predict_gcn(model, X, ei, ew))
    for key, value in checkpoint.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_test_labels_cannot_change_training_or_predictions():
    panel, snapshots, stocks = panel_fixture()
    train, val, test = panel.iloc[:8], panel.iloc[8:16], panel.iloc[16:]
    cfg = {"training": {"gcn": {"hidden_dims": [4], "epochs": 2, "dropout": 0.1}}}
    set_seed(11)
    _, first_model, first_probs = train_gcn(train, val, test, snapshots, snapshots, snapshots, ["x"], stocks, cfg)
    changed = test.copy()
    changed["y_dir"] = 1 - changed["y_dir"]
    set_seed(11)
    _, second_model, second_probs = train_gcn(train, val, changed, snapshots, snapshots, snapshots, ["x"], stocks, cfg)
    np.testing.assert_array_equal(first_probs, second_probs)
    for key, value in first_model.state_dict().items():
        torch.testing.assert_close(value, second_model.state_dict()[key], rtol=0, atol=0)


def test_ece_includes_predictions_exactly_zero():
    assert compute_ece(np.array([1, 0]), np.array([0.0, 1.0])) == 1.0


def test_top_hit_is_positive_return_rate_not_direction_label():
    dates = pd.DatetimeIndex(["2020-01-01"] * 20)
    assert top_decile_hit_rate(dates, np.zeros(20), np.arange(20) / 20, np.ones(20)) == 1.0


def test_json_uses_null_for_undefined_metrics(tmp_path):
    path = tmp_path / "metrics.json"
    save_json({"metric": float("nan"), "nested": [np.float64("inf")]}, path)
    assert json.loads(path.read_text()) == {"metric": None, "nested": [None]}
