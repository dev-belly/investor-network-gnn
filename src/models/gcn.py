"""Lightweight GCN classifier using PyTorch Geometric.

The model has 2 GCN layers + a final MLP head.  Edges carry **weights** which
the GCNConv layer does not natively support; we therefore implement the
propagation manually with ``D^{-1/2} A D^{-1/2}`` so the model remains
*exactly* the GCN of Kipf & Welling (2017).

Inputs
------
- ``x``          : ``(n, d)`` node feature tensor.
- ``edge_index`` : ``(2, E)`` int64 tensor of edges.
- ``edge_weight``: ``(E,)`` float32 tensor of edge weights (post
  symmetric-normalisation, computed in :func:`src.graphs.builders.normalize_adj`).

Output
------
- ``logits``     : ``(n, n_classes)`` classification logits.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sparse aggregation (segment-sum) without depending on torch_scatter."""
    out = torch.zeros(dim_size, src.shape[1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp(min=1.0).unsqueeze(-1)
    summed = scatter_add(src, index, dim_size)
    return summed / counts


class GCNLayer(nn.Module):
    """One GCN layer with symmetric normalisation and edge weights.

    The propagation is::

        H' = sigma(D^{-1/2} (A + I) D^{-1/2} H W)

    where ``A`` is the weighted adjacency.  Self-loops are added at this layer
    so the caller can pass a pure interaction graph.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.lin.weight.size(1))
        self.lin.weight.data.uniform_(-stdv, stdv)
        if self.lin.bias is not None:
            self.lin.bias.data.uniform_(-stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        n = x.shape[0]
        # Add self-loops with weight 1
        self_idx = torch.arange(n, device=x.device)
        ei = torch.cat([edge_index, torch.stack([self_idx, self_idx], dim=0)], dim=1)
        ew = torch.cat([edge_weight, torch.ones(n, device=x.device, dtype=edge_weight.dtype)])

        # Symmetric normalisation: D^{-1/2} A D^{-1/2}
        deg = torch.zeros(n, device=x.device, dtype=edge_weight.dtype)
        deg.index_add_(0, ei[1], ew)
        deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)

        norm_w = ew * deg_inv_sqrt[ei[0]] * deg_inv_sqrt[ei[1]]
        # Aggregate: A_norm @ X @ W
        x_w = self.lin(x)  # (n, out)
        msg = x_w[ei[0]] * norm_w.unsqueeze(-1)  # (E, out)
        out = scatter_add(msg, ei[1], dim_size=n)
        return out


class GCNClassifier(nn.Module):
    """Stack of GCN layers followed by a small MLP head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] | None = None,
        n_classes: int = 2,
        dropout: float = 0.4,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(GCNLayer(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        # Apply GCN layers
        for layer in self.body:
            if isinstance(layer, GCNLayer):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return self.head(x)


def fit_gcn(
    X: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    ei_val: torch.Tensor,
    ew_val: torch.Tensor,
    *,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.4,
    lr: float = 1e-3,
    epochs: int = 120,
    weight_decay: float = 5e-4,
    patience: int = 15,
    device: str | None = None,
    verbose: bool = False,
) -> GCNClassifier:
    """Train a GCN with a held-out validation graph for early stopping."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GCNClassifier(
        in_dim=X.shape[1], hidden_dims=hidden_dims, dropout=dropout, n_classes=2
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    X = X.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    ei_val = ei_val.to(device)
    ew_val = ew_val.to(device)

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    bad = 0

    for epoch in range(epochs):
        model.train()
        optim.zero_grad()
        logits = model(X, edge_index, edge_weight)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val, ei_val, ew_val)
            val_loss = F.cross_entropy(val_logits, y_val).item()
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and (epoch + 1) % 10 == 0:
            log.info(
                "  GCN epoch %d / %d  val_loss=%.4f  val_acc=%.4f",
                epoch + 1,
                epochs,
                val_loss,
                val_acc,
            )
        if bad >= patience:
            if verbose:
                log.info("  Early stopping at epoch %d", epoch + 1)
            break

    model.load_state_dict(best_state)
    return model


def predict_gcn(
    model: GCNClassifier,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    with torch.no_grad():
        logits = model(X, edge_index, edge_weight)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


# -----------------------------------------------------------------------------#
# Optional PyG backend (used only if torch_geometric is installed)
# -----------------------------------------------------------------------------#
class PyGGCNClassifier(nn.Module):
    """Same as :class:`GCNClassifier` but built on torch_geometric.

    Provided as a drop-in if the user installs ``torch_geometric``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] | None = None,
        n_classes: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GCNConv

        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        prev = in_dim
        for h in hidden_dims:
            self.convs.append(GCNConv(prev, h, improved=False, cached=False))
            self.bns.append(nn.BatchNorm1d(h))
            prev = h
        self.dropout = dropout
        self.head = nn.Linear(prev, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from torch_geometric.utils import add_self_loops, degree

        for conv, bn in zip(self.convs, self.bns):
            ei, ew = add_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=x.shape[0])
            deg = degree(ei[1], num_nodes=x.shape[0])
            deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
            ew = ew * deg_inv_sqrt[ei[0]] * deg_inv_sqrt[ei[1]]
            x = conv(x, ei, ew)
            x = bn(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)
