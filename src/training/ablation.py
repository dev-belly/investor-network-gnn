"""Ablation helpers: feature groups, edge drops, graph randomization.

Goal of this module is to answer, for the user, **where does the lift come
from?**  Concretely:

1. **Feature ablation** — drop return/volatility/momentum cluster, drop
   fundamentals, drop network-statistics cluster; rest should fall.
2. **Edge drop**      — at inference time, randomly drop a fraction of edges
   from the test graph and measure how AUC decays.
3. **Graph randomization** — replace the test graph with a randomly
   rewired version (degree-preserving) and check whether the GCN regresses
   to the MLP baseline.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from src.utils.common import get_logger

log = get_logger(__name__)


@dataclass
class FeatureGroups:
    """Convenience grouping used for feature ablation."""

    return_vol: list[str]
    fundamental: list[str]
    momentum: list[str]
    network_stat: list[str]
    other: list[str]


def partition_features(feature_names: list[str]) -> FeatureGroups:
    """Bucket the engineered feature columns by prefix."""
    groups = FeatureGroups(
        return_vol=[c for c in feature_names if any(s in c for s in ("ret_mean", "vol_", "abs_ret_", "skew", "kurt"))],
        fundamental=[c for c in feature_names if c.startswith(("log_market_cap", "beta", "roe", "asset_growth", "leverage", "pe_ttm", "pb", "earnings_yield", "turnover"))],
        momentum=[c for c in feature_names if c.startswith("momentum_")],
        network_stat=[c for c in feature_names if c.startswith(("degree", "eigenvector", "pagerank", "clustering", "core_number", "community"))],
        other=[],
    )
    used = set(groups.return_vol + groups.fundamental + groups.momentum + groups.network_stat)
    groups.other = [c for c in feature_names if c not in used]
    return groups


def random_drop_edges(
    edge_index: np.ndarray, edge_weight: np.ndarray, drop_rate: float, *, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly drop ``drop_rate`` fraction of unique undirected edges."""
    if edge_index.size == 0 or drop_rate <= 0:
        return edge_index, edge_weight
    n_edges = edge_index.shape[1]
    # We assume the edges are duplicated (undirected); keep unique pairs
    pairs = np.stack(
        [np.minimum(edge_index[0], edge_index[1]), np.maximum(edge_index[0], edge_index[1])],
        axis=1,
    )
    unique_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
    n_unique = unique_pairs.shape[0]
    n_drop = int(round(drop_rate * n_unique))
    if n_drop <= 0:
        return edge_index, edge_weight
    drop_idx = rng.choice(n_unique, size=n_drop, replace=False)
    keep_mask = np.ones(n_unique, dtype=bool)
    keep_mask[drop_idx] = False
    keep = np.isin(inv, np.where(keep_mask)[0])
    return edge_index[:, keep], edge_weight[keep]


def randomize_graph(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    n_nodes: int,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Degree-preserving edge swap (Configuration model).

    Produces a randomized baseline that preserves the *degree sequence* but
    destroys any community / assortativity / topology.  If the GCN still
    works on this randomized graph, the lift is not coming from network
    relationships.
    """
    if edge_index.size == 0:
        return edge_index, edge_weight
    # Build the undirected graph and use NetworkX's double_edge_swap
    pairs = np.stack(
        [np.minimum(edge_index[0], edge_index[1]),
         np.maximum(edge_index[0], edge_index[1])],
        axis=1,
    )
    unique_pairs = np.unique(pairs, axis=0)
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for u, v in unique_pairs:
        g.add_edge(int(u), int(v))
    if g.number_of_edges() < 2:
        return edge_index, edge_weight
    try:
        # 5x edges random swaps is a decent default
        n_swap = 5 * g.number_of_edges()
        g_rand = nx.double_edge_swap(g, nswap=n_swap, max_tries=10 * n_swap)
    except nx.NetworkXAlgorithmError:
        return edge_index, edge_weight
    # Re-attach original weights randomly
    src = []
    dst = []
    w = []
    edges = list(g_rand.edges())
    if not edges:
        return edge_index, edge_weight
    perm = rng.permutation(len(edge_weight) // 2 if edge_weight.size > 0 else 0)
    for i, (u, v) in enumerate(edges):
        src.append(u); dst.append(v)
        if edge_weight.size > 0:
            w.append(edge_weight[perm[i % len(perm)]] if len(perm) else 1.0)
        else:
            w.append(1.0)
    src = np.array(src); dst = np.array(dst); w = np.array(w, dtype=np.float32)
    src_full = np.concatenate([src, dst]); dst_full = np.concatenate([dst, src])
    w_full = np.concatenate([w, w])
    return np.stack([src_full, dst_full], axis=0), w_full
