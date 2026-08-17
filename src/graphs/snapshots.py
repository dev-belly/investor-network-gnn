"""Time-aware graph snapshot building."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.graphs.builders import (
    fund_holding_graph,
    industry_prior_graph,
    knn_graph,
    threshold_graph,
    union_graph,
)
from src.graphs.stats import add_industry_community, network_statistics
from src.utils.common import get_logger

log = get_logger(__name__)


@dataclass
class GraphSnapshot:
    """All graph-related artefacts at a given date t."""

    date: pd.Timestamp
    edge_index: np.ndarray       # (2, E)
    edge_weight: np.ndarray      # (E,)
    network_stats: pd.DataFrame  # (n, k)
    method: str

    def edge_count(self) -> int:
        if self.edge_index.size == 0:
            return 0
        return int(self.edge_index.shape[1] // 2)


def build_snapshot(
    returns: pd.DataFrame,
    fund_holdings: pd.DataFrame,
    stock_industry: pd.Series,
    date: pd.Timestamp,
    *,
    lookback: int = 60,
    method: str = "union",
    threshold_quantile: float = 0.85,
    knn_k: int = 10,
    fund_overlap_quantile: float = 0.85,
) -> GraphSnapshot:
    """Build a single-date graph snapshot using only information up to ``date``.

    The temporal split ensures no leakage: labels at ``date`` predict the
    *next* horizon, so the graph sees only past returns.

    Returns an object with edge_index/edge_weight plus network statistics.
    """
    n = returns.shape[1]
    stock_ids = list(returns.columns)
    date_loc = returns.index.get_loc(date)
    start_loc = max(0, date_loc - lookback)
    window = returns.iloc[start_loc:date_loc]

    industry_index = stock_industry.reindex(stock_ids)
    industry_ei, industry_w = industry_prior_graph(industry_index)

    graphs = []
    if method in ("threshold", "union"):
        thr_ei, thr_w = threshold_graph(window, quantile=threshold_quantile, use_absolute=True)
        graphs.append((thr_ei, thr_w))
    if method in ("knn", "union"):
        knn_ei, knn_w = knn_graph(window, k=knn_k, use_absolute=True)
        graphs.append((knn_ei, knn_w))
    if method in ("union", "industry", "fund"):
        graphs.append((industry_ei, industry_w))
    if method in ("union", "fund"):
        fund_ei, fund_w = fund_holding_graph(
            fund_holdings, n_stocks=n, overlap_quantile=fund_overlap_quantile
        )
        graphs.append((fund_ei, fund_w))

    if method == "single":
        edge_index, edge_weight = threshold_graph(window, quantile=threshold_quantile)
    else:
        edge_index, edge_weight = union_graph(graphs, n)

    net_stats = network_statistics(edge_index, edge_weight, n)
    net_stats = add_industry_community(net_stats, stock_industry.reindex(stock_ids))

    return GraphSnapshot(
        date=date,
        edge_index=edge_index,
        edge_weight=edge_weight,
        network_stats=net_stats,
        method=method,
    )


def build_snapshot_series(
    returns: pd.DataFrame,
    fund_holdings: pd.DataFrame,
    stock_industry: pd.Series,
    dates: list[pd.Timestamp],
    **kwargs,
) -> dict[pd.Timestamp, GraphSnapshot]:
    """Build snapshots for many dates, logging progress."""
    snapshots: dict[pd.Timestamp, GraphSnapshot] = {}
    for i, d in enumerate(dates):
        if i % 10 == 0:
            log.info("  Snapshot %d/%d -> %s", i + 1, len(dates), d.date())
        try:
            snapshots[d] = build_snapshot(
                returns=returns,
                fund_holdings=fund_holdings,
                stock_industry=stock_industry,
                date=d,
                **kwargs,
            )
        except Exception as e:  # pragma: no cover
            log.warning("Failed to build snapshot for %s: %s", d, e)
    return snapshots
