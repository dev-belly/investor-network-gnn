"""Stock graph builders.

Each function returns an ``(edge_index, edge_weight)`` numpy pair:

- ``edge_index`` : ``(2, E)`` int64 array, undirected (both directions).
- ``edge_weight`` : ``(E,)`` float32 array, normalized to ``[0, 1]``.

The graph is constructed *for a single date* using information up to ``t``.
This module is deliberately side-effect free; callers orchestrate the time
axis.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def symmetrize(
    edge_index: np.ndarray, edge_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure the edge set is undirected (a-b and b-a both present)."""
    if edge_index.size == 0:
        return edge_index.astype(np.int64), edge_weight.astype(np.float32)
    pairs = np.stack([np.minimum(edge_index[0], edge_index[1]),
                      np.maximum(edge_index[0], edge_index[1])], axis=1)
    # Average duplicate weights
    df = pd.DataFrame({"w": edge_weight, "i": pairs[:, 0], "j": pairs[:, 1]})
    df = df.groupby(["i", "j"], as_index=False)["w"].mean()
    src = df["i"].values
    dst = df["j"].values
    out_index = np.stack([src, dst], axis=0)
    out_index_rev = np.stack([dst, src], axis=0)
    full_index = np.concatenate([out_index, out_index_rev], axis=1)
    full_weight = np.concatenate([df["w"].values, df["w"].values])
    return full_index.astype(np.int64), full_weight.astype(np.float32)


def normalize_adj(edge_index: np.ndarray, edge_weight: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric normalization D^{-1/2} (A) D^{-1/2}.

    Self-loops are added so each node has at least one neighbor (itself).
    """
    src = edge_index[0]
    dst = edge_index[1]

    # Add self-loops
    self_idx = np.arange(n)
    src = np.concatenate([src, self_idx])
    dst = np.concatenate([dst, self_idx])
    w = np.concatenate([edge_weight, np.ones(n)])

    deg = np.bincount(dst, weights=w, minlength=n)
    deg_inv_sqrt = np.zeros(n, dtype=np.float32)
    mask = deg > 0
    deg_inv_sqrt[mask] = np.power(deg[mask], -0.5)
    norm_w = w * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    return edge_index, norm_w.astype(np.float32)


# -----------------------------------------------------------------------------#
# Builders
# -----------------------------------------------------------------------------#
def threshold_graph(
    returns_window: pd.DataFrame,
    quantile: float = 0.8,
    use_absolute: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a graph where edges connect highly-correlated pairs.

    Parameters
    ----------
    returns_window : pd.DataFrame
        Returns over the lookback window, indexed by date (cols = stocks).
    quantile : float
        Keep edges whose pairwise correlation is above this quantile.
    use_absolute : bool
        If True (default), keep strong |corr|.  If False keep strong positive corr only.
    """
    corr = returns_window.corr().fillna(0.0)
    n = corr.shape[0]
    if use_absolute:
        scores = np.abs(corr.values).astype(np.float64).copy()
    else:
        scores = corr.values.astype(np.float64).copy()
    np.fill_diagonal(scores, 0.0)

    # Threshold = empirical quantile of off-diagonal entries (|.| or raw)
    off_diag = scores[np.triu_indices(n, k=1)]
    if len(off_diag) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    thr = np.quantile(off_diag, quantile)
    mask = scores >= thr
    rows, cols = np.where(np.triu(mask, k=1))
    if len(rows) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    weights = scores[rows, cols]
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge_index = np.stack([src, dst], axis=0)
    # Map weights to [0, 1]
    if use_absolute:
        # The threshold value is the upper quantile of |corr|, weights are |corr|
        # Normalize to (thr, 1) -> (0, 1)
        w = (weights - thr) / max(1e-9, 1.0 - thr)
    else:
        w = (weights - thr) / max(1e-9, 1.0 - thr)
    w = np.clip(w, 0.0, 1.0)
    if np.any(weights > 0) and not np.any(w > 0):
        # Equal positive correlations still represent real graph links.
        w = np.ones_like(w)
    w = np.concatenate([w, w])
    return edge_index.astype(np.int64), w.astype(np.float32)


def knn_graph(
    returns_window: pd.DataFrame,
    k: int = 10,
    use_absolute: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """k-nearest neighbor graph by correlation distance."""
    corr = returns_window.corr().fillna(0.0)
    n = corr.shape[0]
    if use_absolute:
        scores = np.abs(corr.values).astype(np.float64).copy()
    else:
        scores = corr.values.astype(np.float64).copy()
    np.fill_diagonal(scores, -np.inf)
    # For each node, keep top-k neighbors
    edge_src = []
    edge_dst = []
    edge_w = []
    for i in range(n):
        top = np.argpartition(-scores[i], kth=min(k, n - 1))[:k]
        for j in top:
            if i == j:
                continue
            edge_src.append(i)
            edge_dst.append(j)
            edge_w.append(scores[i, j])
    if not edge_src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    edge_index = np.stack([np.array(edge_src), np.array(edge_dst)], axis=0)
    edge_w = np.array(edge_w, dtype=np.float32)
    edge_index, edge_w = symmetrize(edge_index, edge_w)
    if edge_w.max() > edge_w.min():
        edge_w = (edge_w - edge_w.min()) / (edge_w.max() - edge_w.min())
    elif edge_w.max() > 0:
        # A uniform positive correlation must still transmit messages.
        edge_w = np.ones_like(edge_w)
    return edge_index.astype(np.int64), edge_w.astype(np.float32)


def industry_prior_graph(
    stock_industry: pd.Series,
    intra_industry_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Industry prior: edges connect every pair within the same industry."""
    # Node positions must retain the order supplied by the returns panel.
    src = []
    dst = []
    for industry, group in stock_industry.groupby(stock_industry):
        ids = group.index.tolist()
        id_map = {s: i for i, s in enumerate(stock_industry.index)}
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                src.append(id_map[ids[i_idx]])
                dst.append(id_map[ids[j_idx]])
    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    src = np.array(src)
    dst = np.array(dst)
    w = np.full(len(src), intra_industry_weight, dtype=np.float32)
    # undirected
    src_full = np.concatenate([src, dst])
    dst_full = np.concatenate([dst, src])
    w_full = np.concatenate([w, w])
    return np.stack([src_full, dst_full], axis=0).astype(np.int64), w_full


def fund_holding_graph(
    fund_holdings: pd.DataFrame,
    n_stocks: int,
    overlap_quantile: float = 0.85,
    stock_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a graph connecting stocks frequently co-held by funds.

    Edge weight = number of funds that hold both endpoints.  Edges below the
    quantile threshold (default 85th pct) are dropped to keep the graph
    sparse.
    """
    # Pivot fund x stock weight
    pivot = (
        fund_holdings.assign(w=fund_holdings["weight"])
        .pivot_table(index="fund_id", columns="stock_id", values="w", fill_value=0.0)
    )
    if stock_ids is None:
        raise ValueError("stock_ids are required to align holdings with graph nodes")
    ordered_ids = list(stock_ids)
    if len(ordered_ids) != n_stocks or len(set(ordered_ids)) != n_stocks:
        raise ValueError("stock_ids must contain each graph node exactly once")
    # Pivot sorts its columns and omits stocks with no fund holding; both
    # would otherwise make the resulting edge positions refer to wrong nodes.
    pivot = pivot.reindex(columns=ordered_ids, fill_value=0.0)
    # Co-occurrence matrix: dot product of indicators
    indicator = (pivot > 0).astype(np.int32)
    co = indicator.values.T @ indicator.values  # (n_stocks_used, n_stocks_used)

    np.fill_diagonal(co, 0)
    # threshold by quantile
    off_diag = co[np.triu_indices_from(co, k=1)]
    if len(off_diag) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    thr = np.quantile(off_diag, overlap_quantile)
    rows, cols = np.where(np.triu(co >= max(thr, 1), k=1))
    if len(rows) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    weights = co[rows, cols].astype(np.float32)
    if weights.max() > weights.min():
        weights = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        # With one edge (or equal overlap counts), min-max scaling would turn
        # every real co-holding relation into a zero-weight edge.
        weights = np.ones_like(weights)
    edge_index = np.stack([src, dst], axis=0)
    return edge_index.astype(np.int64), np.concatenate([weights, weights]).astype(np.float32)


def union_graph(
    graphs: Sequence[tuple[np.ndarray, np.ndarray]],
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine multiple graphs by taking the **union** of edges and averaging
    weights.  Useful as the default input to the GCN model."""
    all_src = []
    all_dst = []
    all_w = []
    for ei, w in graphs:
        if ei.size == 0:
            continue
        all_src.append(ei[0])
        all_dst.append(ei[1])
        all_w.append(w)
    if not all_src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    w = np.concatenate(all_w)
    df = pd.DataFrame({"s": src, "d": dst, "w": w})
    df = df.groupby(["s", "d"], as_index=False)["w"].mean()
    return df[["s", "d"]].values.T.astype(np.int64), df["w"].values.astype(np.float32)
