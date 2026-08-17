"""Network statistics computed from a graph.

These are per-node scalar features that capture *structural* signals without
training a GNN.  Adding them to the feature matrix lets us test whether the
**graph structure itself** (rather than just the message passing) carries
information.

Computed features
-----------------
- ``degree``             : weighted degree
- ``degree_cent``        : normalized degree centrality
- ``eigenvector_cent``   : eigenvector centrality (NetworkX)
- ``pagerank``           : PageRank (NetworkX)
- ``clustering``         : weighted clustering coefficient
- ``core_number``        : k-core decomposition
- ``community_id``       : Louvain community label (categorical -> one-hot later)
- ``community_size``     : size of the community
"""
from __future__ import annotations

import warnings
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain  # python-louvain
from scipy.sparse import coo_matrix


def _edge_to_nx(
    edge_index: np.ndarray, edge_weight: np.ndarray, n: int
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if edge_index.size == 0:
        return g
    for (u, v), w in zip(edge_index.T, edge_weight):
        if u == v:
            continue
        # If duplicate, keep max weight
        if g.has_edge(u, v):
            if g[u][v]["weight"] < w:
                g[u][v]["weight"] = float(w)
        else:
            g.add_edge(int(u), int(v), weight=float(w))
    return g


def network_statistics(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    n: int,
    *,
    stock_industry: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute per-node network statistics.

    Returns a ``(n, k)`` DataFrame indexed by node id.
    """
    stats: dict[str, np.ndarray] = {}

    # ---- Degree -------------------------------------------------------------
    if edge_index.size == 0:
        deg = np.zeros(n, dtype=np.float32)
    else:
        deg = np.bincount(edge_index[0], weights=edge_weight, minlength=n)
    stats["degree"] = deg
    stats["degree_cent"] = deg / max(1.0, deg.max())

    # ---- Build NX graph for heavy lifting -----------------------------------
    g = _edge_to_nx(edge_index, edge_weight, n)

    # ---- Eigenvector centrality (may fail on disconnected graphs) -----------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ev = nx.eigenvector_centrality_numpy(g, weight="weight")
            stats["eigenvector_cent"] = np.array([ev[i] for i in range(n)], dtype=np.float32)
        except Exception:
            stats["eigenvector_cent"] = np.zeros(n, dtype=np.float32)

    # ---- PageRank -----------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pr = nx.pagerank(g, weight="weight", alpha=0.85, max_iter=200)
            stats["pagerank"] = np.array([pr[i] for i in range(n)], dtype=np.float32)
        except Exception:
            stats["pagerank"] = np.zeros(n, dtype=np.float32)

    # ---- Weighted clustering coefficient ------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cl = nx.clustering(g, weight="weight")
            stats["clustering"] = np.array([cl[i] for i in range(n)], dtype=np.float32)
        except Exception:
            stats["clustering"] = np.zeros(n, dtype=np.float32)

    # ---- K-core -------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            core = nx.core_number(g)
            stats["core_number"] = np.array([core[i] for i in range(n)], dtype=np.float32)
        except Exception:
            stats["core_number"] = np.zeros(n, dtype=np.float32)

    # ---- Community detection (Louvain) --------------------------------------
    try:
        if edge_index.size > 0:
            partition = community_louvain.best_partition(
                _to_undirected_nx(edge_index, edge_weight, n), weight="weight", random_state=0
            )
            comm = np.array([partition[i] for i in range(n)], dtype=np.int32)
        else:
            comm = np.zeros(n, dtype=np.int32)
    except Exception:
        comm = np.zeros(n, dtype=np.int32)
    stats["community_id"] = comm
    sizes = pd.Series(comm).value_counts().to_dict()
    stats["community_size"] = np.array([sizes[c] for c in comm], dtype=np.float32)

    df = pd.DataFrame(stats)
    df.index.name = "node"
    return df


def _to_undirected_nx(
    edge_index: np.ndarray, edge_weight: np.ndarray, n: int
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if edge_index.size == 0:
        return g
    for (u, v), w in zip(edge_index.T, edge_weight):
        if u == v:
            continue
        if g.has_edge(u, v):
            g[u][v]["weight"] = max(g[u][v]["weight"], float(w))
        else:
            g.add_edge(int(u), int(v), weight=float(w))
    return g


def add_industry_community(
    network_stats: pd.DataFrame, stock_industry: pd.Series
) -> pd.DataFrame:
    """Append industry as an additional 'community' baseline.

    The idea is that one might naively cluster stocks by SW industry.  We make
    that explicit so the model can compare learned communities vs. official
    ones.
    """
    industry = stock_industry.reset_index(drop=True)
    out = network_stats.copy()
    out["industry_id"] = pd.Categorical(industry).codes.astype(np.int32)
    return out
