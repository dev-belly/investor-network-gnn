"""Baseline models: Logistic Regression, XGBoost, MLP.

All baselines are *cross-sectional*: they consume a flat feature vector per
``(date, stock_id)`` sample and emit a probability.  Crucially, the MLP
baseline **does not** use graph structure, which makes it the apples-to-apples
reference point for the GCN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.common import get_logger

log = get_logger(__name__)


# -----------------------------------------------------------------------------#
# Classical baselines
# -----------------------------------------------------------------------------#
@dataclass
class ClassicalPredictions:
    probs: np.ndarray
    model: object


def fit_logreg(
    X_train: np.ndarray, y_train: np.ndarray, *, C: float = 1.0
) -> LogisticRegression:
    log.info("Fitting Logistic Regression ...")
    model = LogisticRegression(
        C=C, solver="lbfgs", max_iter=2000
    )
    model.fit(X_train, y_train)
    return model


def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 400,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> XGBClassifier:
    log.info(
        "Fitting XGBoost (n_est=%d, depth=%d, lr=%.3f) ...",
        n_estimators,
        max_depth,
        learning_rate,
    )
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        n_jobs=2,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------------------------------------------------------#
# MLP baseline (no graph, just feature MLPs)
# -----------------------------------------------------------------------------#
class MLPClassifier(nn.Module):
    """Simple feedforward classifier with batch norm and dropout.

    This is the *non-graph* neural baseline.  Comparison against the GCN
    isolates the marginal value of message passing.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] | None = None,
        n_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 80,
    batch_size: int = 1024,
    device: str | None = None,
    verbose: bool = False,
) -> MLPClassifier:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = MLPClassifier(
        in_dim=X_train_s.shape[1], hidden_dims=hidden_dims, dropout=dropout
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    Xt = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    best_val_acc = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    n = Xt.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            optim.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_acc = (val_logits.argmax(dim=1) == yv).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch + 1) % 10 == 0:
            log.info("  MLP epoch %d / %d  val_acc=%.4f", epoch + 1, epochs, val_acc)

    model.load_state_dict(best_state)
    # Stash the scaler so the caller can transform inference inputs
    model._scaler = scaler  # type: ignore[attr-defined]
    return model


def predict_mlp(model: MLPClassifier, X: np.ndarray, device: str | None = None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = getattr(model, "_scaler", StandardScaler().fit(X))
    Xs = scaler.transform(X)
    Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(Xt)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs
