"""Unit tests for graph builders and feature engineering.

Run with::

    .venv/bin/python -m pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import compute_node_features, make_labels
from src.data.synthetic import generate_universe
from src.graphs.builders import (
    fund_holding_graph,
    industry_prior_graph,
    knn_graph,
    normalize_adj,
    threshold_graph,
    union_graph,
)
from src.graphs.stats import network_statistics
from src.training.ablation import random_drop_edges, randomize_graph
from src.training.metrics import compute_auc, compute_brier, compute_ece
from src.training.split import chronological_split, filter_by_split


# -----------------------------------------------------------------------------#
# Fixtures
# -----------------------------------------------------------------------------#
@pytest.fixture(scope="module")
def tiny_universe():
    return generate_universe(
        n_stocks=60, n_days=400, n_industries=10, n_funds=30, start_date="2020-01-01", seed=7
    )


# -----------------------------------------------------------------------------#
# Graph builders
# -----------------------------------------------------------------------------#
def test_threshold_graph_shapes(tiny_universe):
    returns = tiny_universe.returns.iloc[-120:]
    ei, w = threshold_graph(returns, quantile=0.85)
    assert ei.shape[0] == 2
    # Undirected: each unique (min, max) pair appears exactly twice
    pairs = np.stack(
        [np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])], axis=1
    )
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    assert (counts == 2).all(), "threshold graph should be undirected"
    assert w.shape[0] == ei.shape[1]
    assert (w >= 0).all() and (w <= 1).all()


def test_knn_graph_symmetric(tiny_universe):
    returns = tiny_universe.returns.iloc[-120:]
    ei, w = knn_graph(returns, k=5)
    pairs = np.stack([np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])], axis=1)
    # Each unique pair appears exactly twice (undirected)
    df = pd.DataFrame(pairs, columns=["a", "b"])
    counts = df.value_counts()
    assert (counts == 2).all()


def test_industry_prior_graph_complete(tiny_universe):
    ei, w = industry_prior_graph(tiny_universe.stock_industry)
    n = len(tiny_universe.stock_industry)
    # Every pair within the same industry is connected
    assert ei.shape[0] == 2
    assert w.shape[0] == ei.shape[1]
    # Total undirected unique edges = sum_{ind} n_i(n_i-1)/2
    unique_undirected = 0
    for _, g in tiny_universe.stock_industry.groupby(tiny_universe.stock_industry):
        ni = len(g)
        unique_undirected += ni * (ni - 1) // 2
    # The implementation emits each undirected edge twice (u->v and v->u)
    assert ei.shape[1] == 2 * unique_undirected
    assert (w == 1.0).all()


def test_union_graph_handles_empty(tiny_universe):
    ei, w = union_graph([], n_nodes=10)
    assert ei.shape == (2, 0)
    assert w.shape == (0,)


def test_normalize_adj_self_loop_added(tiny_universe):
    returns = tiny_universe.returns.iloc[-60:]
    ei, w = threshold_graph(returns, quantile=0.95)
    ei_n, w_n = normalize_adj(ei, w, n=returns.shape[1])
    # After self-loop addition, total degree (counted) should grow by 1 per node
    n = returns.shape[1]
    n_self_edges = (ei_n[0] == ei_n[1]).sum()
    assert n_self_edges == n  # one self-loop per node


def test_network_statistics_keys(tiny_universe):
    returns = tiny_universe.returns.iloc[-60:]
    ei, w = threshold_graph(returns, quantile=0.85)
    df = network_statistics(ei, w, n=returns.shape[1])
    expected = {"degree", "degree_cent", "eigenvector_cent", "pagerank",
                "clustering", "core_number", "community_id", "community_size"}
    assert set(df.columns) >= expected


# -----------------------------------------------------------------------------#
# Ablation helpers
# -----------------------------------------------------------------------------#
def test_random_drop_edges_preserves_count_when_zero():
    rng = np.random.default_rng(0)
    ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    w = np.array([0.5, 0.5, 0.3, 0.3])
    ei2, w2 = random_drop_edges(ei, w, 0.0, rng=rng)
    assert ei2.shape == ei.shape
    np.testing.assert_array_equal(ei, ei2)


def test_random_drop_edges_drops_some():
    rng = np.random.default_rng(0)
    ei = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    w = np.array([0.5, 0.5, 0.3, 0.3, 0.4, 0.4])
    ei2, w2 = random_drop_edges(ei, w, 0.5, rng=rng)
    assert ei2.shape[1] < ei.shape[1]


def test_randomize_graph_preserves_degree():
    rng = np.random.default_rng(0)
    # Build a small chain graph
    ei = np.array(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]]
    )
    w = np.ones(ei.shape[1])
    # Degree sequence of original
    n = 5
    deg_orig = np.bincount(ei[0], minlength=n)
    ei2, _ = randomize_graph(ei, w, n_nodes=n, rng=rng)
    deg_rand = np.bincount(ei2[0], minlength=n)
    np.testing.assert_array_equal(deg_orig, deg_rand)


# -----------------------------------------------------------------------------#
# Features & labels
# -----------------------------------------------------------------------------#
def test_feature_panel_shape(tiny_universe):
    cfg = {
        "features": {
            "return_windows": [5, 20],
            "volatility_windows": [20],
            "momentum_windows": [20, 60],
            "fundamental_n": 8,
        },
        "label": {"horizon": 20, "upper_quantile": 0.5, "risk_quantiles": [0.2, 0.8]},
    }
    feats, names = compute_node_features(tiny_universe.returns, tiny_universe.fundamentals, cfg)
    assert isinstance(feats, pd.DataFrame)
    assert feats.index.names == ["date", "stock_id"]
    assert len(names) == feats.shape[1]
    assert not feats.isna().any().any()
    assert (feats.select_dtypes(include="number").shape[1] == feats.shape[1])


def test_labels_balance(tiny_universe):
    cfg = {
        "label": {"horizon": 20, "upper_quantile": 0.5, "risk_quantiles": [0.2, 0.8]},
    }
    labels = make_labels(tiny_universe.returns, cfg)
    assert set(["y_dir", "y_risk", "fwd_ret"]).issubset(set(labels.columns))
    # y_dir should be roughly balanced
    assert 0.45 < labels["y_dir"].mean() < 0.55
    # y_risk should cover all 3 classes
    assert set(labels["y_risk"].unique()) == {0, 1, 2}


# -----------------------------------------------------------------------------#
# Time split
# -----------------------------------------------------------------------------#
def test_chronological_split_purge_gap():
    dates = pd.bdate_range("2020-01-01", periods=300)
    split = chronological_split(dates, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, purge_gap=5)
    # train and val should not overlap
    assert split.train[1] < split.val[0]
    # purge_gap should be respected
    gap = (split.val[0] - split.train[1]).days
    assert gap >= 5


def test_filter_by_split(tiny_universe):
    dates = pd.bdate_range("2020-01-01", periods=300)
    split = chronological_split(dates)
    panel = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [dates, ["S0", "S1"]], names=["date", "stock_id"]
        ),
        data={"x": np.random.randn(len(dates) * 2)},
    )
    train = filter_by_split(panel, split, "train")
    val = filter_by_split(panel, split, "val")
    assert (train.index.get_level_values("date") <= split.train[1]).all()
    assert (val.index.get_level_values("date") >= split.val[0]).all()


# -----------------------------------------------------------------------------#
# Metrics
# -----------------------------------------------------------------------------#
def test_metrics_basic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    p = rng.uniform(0, 1, size=1000)
    auc = compute_auc(y, p)
    brier = compute_brier(y, p)
    ece = compute_ece(y, p, n_bins=10)
    assert 0.4 < auc < 0.6  # random-ish
    assert 0.2 < brier < 0.4
    assert 0.0 <= ece <= 0.5
