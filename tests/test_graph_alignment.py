"""Graph edge positions must describe the same stocks as the model feature rows."""

import numpy as np
import pandas as pd
import pytest

from src.graphs.builders import fund_holding_graph, industry_prior_graph, knn_graph, threshold_graph
from src.graphs.snapshots import build_snapshot


def undirected_pairs(edge_index):
    return {frozenset(pair) for pair in edge_index.T.tolist()}


def test_industry_edges_keep_returns_column_order():
    industries = pd.Series(["tech", "finance", "tech"], index=["C", "A", "B"])
    edges, _ = industry_prior_graph(industries)
    assert undirected_pairs(edges) == {frozenset((0, 2))}


def test_fund_edges_keep_returns_column_order_and_missing_stocks():
    holdings = pd.DataFrame({"fund_id": ["F1", "F1"], "stock_id": ["C", "B"], "weight": [0.5, 0.5]})
    edges, _ = fund_holding_graph(holdings, n_stocks=3, stock_ids=["C", "A", "B"])
    assert undirected_pairs(edges) == {frozenset((0, 2))}
    with pytest.raises(ValueError, match="stock_ids are required"):
        fund_holding_graph(holdings, n_stocks=3)
    with pytest.raises(ValueError, match="each graph node exactly once"):
        fund_holding_graph(holdings, n_stocks=3, stock_ids=["C", "C", "B"])


def test_fund_edges_keep_unsorted_order_when_every_stock_is_held():
    holdings = pd.DataFrame(
        {"fund_id": ["F1", "F1", "F2"], "stock_id": ["C", "B", "A"], "weight": [0.5, 0.5, 1.0]}
    )
    edges, _ = fund_holding_graph(holdings, n_stocks=3, stock_ids=["C", "A", "B"])
    assert undirected_pairs(edges) == {frozenset((0, 2))}


def test_fund_snapshot_does_not_mix_in_industry_edges():
    dates = pd.bdate_range("2020-01-01", periods=10)
    returns = pd.DataFrame(
        np.arange(30, dtype=float).reshape(10, 3) / 1000,
        index=dates,
        columns=["C", "A", "B"],
    )
    holdings = pd.DataFrame({"fund_id": ["F1", "F1"], "stock_id": ["C", "B"], "weight": [0.5, 0.5]})
    industries = pd.Series(["tech", "tech", "tech"], index=returns.columns)

    snapshot = build_snapshot(returns, holdings, industries, dates[-1], method="fund")
    assert undirected_pairs(snapshot.edge_index) == {frozenset((0, 2))}


def test_fund_overlap_count_does_not_overflow_at_128_funds():
    funds = [f"F{i}" for i in range(130)]
    holdings = pd.DataFrame(
        {
            "fund_id": [fund for fund in funds for _ in range(2)],
            "stock_id": [stock for _ in funds for stock in ("A", "B")],
            "weight": [0.5] * 260,
        }
    )
    edges, weights = fund_holding_graph(holdings, n_stocks=2, stock_ids=["A", "B"])
    assert undirected_pairs(edges) == {frozenset((0, 1))}
    assert (weights > 0).all()


def test_equal_positive_knn_links_keep_nonzero_weight():
    dates = pd.bdate_range("2020-01-01", periods=4)
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03, 0.04], "B": [0.02, 0.04, 0.06, 0.08]},
        index=dates,
    )
    edges, weights = knn_graph(returns, k=1)
    assert undirected_pairs(edges) == {frozenset((0, 1))}
    assert (weights > 0).all()


def test_equal_positive_threshold_links_keep_nonzero_weight():
    dates = pd.bdate_range("2020-01-01", periods=4)
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03, 0.04], "B": [0.02, 0.04, 0.06, 0.08]},
        index=dates,
    )
    edges, weights = threshold_graph(returns)
    assert undirected_pairs(edges) == {frozenset((0, 1))}
    assert (weights > 0).all()
