"""End-to-end training orchestration.

This module is the *controller* that ties together the data panel, the graph
snapshots, and the model zoo.  It exposes a small surface area so the main
script can run "everything" with one call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.models.baselines import fit_logreg, fit_mlp, fit_xgboost, predict_mlp
from src.models.gcn import fit_gcn, predict_gcn
from src.training.ablation import (
    FeatureGroups,
    partition_features,
    random_drop_edges,
    randomize_graph,
)
from src.training.metrics import EvalResult, evaluate_predictions
from src.training.split import TimeSplit, filter_by_split, select_graph_dates
from src.utils.common import get_logger

log = get_logger(__name__)


@dataclass
class TrainArtifacts:
    """All model outputs plus the evaluation table."""

    eval_results: list[EvalResult]
    model_objects: dict


# -----------------------------------------------------------------------------#
# Baseline panel training
# -----------------------------------------------------------------------------#
def train_classical_baselines(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
) -> tuple[EvalResult, EvalResult, EvalResult, dict]:
    """Train LR + XGBoost + MLP on the cross-sectional panel."""
    Xt = train[feature_cols].values
    yt = train["y_dir"].values.astype(int)
    Xv = val[feature_cols].values
    yv = val["y_dir"].values.astype(int)
    Xe = test[feature_cols].values
    ye = test["y_dir"].values.astype(int)

    models: dict = {}

    # ---- Logistic Regression ------------------------------------------------
    lr_model = fit_logreg(Xt, yt, C=1.0)
    p_lr = lr_model.predict_proba(Xe)[:, 1]
    models["logreg"] = lr_model
    res_lr = evaluate_predictions(
        test.reset_index(), p_lr, name="LR-test",
        industry_series=test.index.get_level_values("stock_id").map(
            test.index.get_level_values("stock_id")
        ),
    )
    # Provide industry using the panel (it carries the multi-index with stock_id)
    res_lr = evaluate_predictions(
        test, p_lr, name="LogReg",
        industry_series=test.index.get_level_values("stock_id"),
    )

    # ---- XGBoost ------------------------------------------------------------
    xgb_cfg = cfg["training"]["baseline"]["xgb"]
    xgb = fit_xgboost(Xt, yt, **xgb_cfg)
    p_xgb = xgb.predict_proba(Xe)[:, 1]
    models["xgboost"] = xgb
    res_xgb = evaluate_predictions(test, p_xgb, name="XGBoost",
        industry_series=test.index.get_level_values("stock_id"))

    # ---- MLP (no graph) -----------------------------------------------------
    mlp_cfg = cfg["training"]["baseline"]["mlp"]
    mlp = fit_mlp(Xt, yt, Xv, yv, **mlp_cfg, verbose=False)
    p_mlp = predict_mlp(mlp, Xe)
    models["mlp"] = mlp
    res_mlp = evaluate_predictions(test, p_mlp, name="MLP-no-graph",
        industry_series=test.index.get_level_values("stock_id"))

    return res_lr, res_xgb, res_mlp, models


# -----------------------------------------------------------------------------#
# GCN training
# -----------------------------------------------------------------------------#
def _snapshots_to_arrays(
    panel: pd.DataFrame,
    snapshots: dict,
    stock_ids: list[str],
    feature_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack (date, stock_id) rows into per-date tensors.

    Each date gets its own (X, edge_index, edge_weight, y).  We pack them
    into a single large "batch" (concatenating nodes across dates) so that
    message passing crosses the entire cross-section per period.  Edge cases
    where a stock drops out of the universe are handled by zero-filling.
    """
    n_stocks = len(stock_ids)
    id_to_idx = {s: i for i, s in enumerate(stock_ids)}
    # Materialise feature matrix per date
    X_blocks = []
    y_blocks = []
    ei_blocks = []
    ew_blocks = []
    # Track date boundaries for inference
    date_ranges: list[tuple[int, int]] = []

    cur_offset = 0
    for date, sub in panel.groupby(level="date"):
        # Order rows by stock_ids
        sub = sub.reindex(stock_ids, level="stock_id")
        feat = sub[feature_cols].values.astype(np.float32)
        # NaN -> 0 will be safe inside the model; better drop but we keep all.
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        X_blocks.append(feat)
        y_blocks.append(sub["y_dir"].values.astype(np.int64))

        snap = snapshots.get(date)
        if snap is not None and snap.edge_index.size > 0:
            src = snap.edge_index[0].copy()
            dst = snap.edge_index[1].copy()
            w = snap.edge_weight.copy()
            # Map node ids (assumed already aligned)
            ei_blocks.append(
                np.stack(
                    [src + cur_offset, dst + cur_offset], axis=0
                ).astype(np.int64)
            )
            ew_blocks.append(w.astype(np.float32))

        date_ranges.append((cur_offset, cur_offset + n_stocks))
        cur_offset += n_stocks

    X = torch.tensor(np.concatenate(X_blocks, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_blocks, axis=0), dtype=torch.long)
    if ei_blocks:
        edge_index = torch.tensor(np.concatenate(ei_blocks, axis=1), dtype=torch.long)
        edge_weight = torch.tensor(np.concatenate(ew_blocks, axis=0), dtype=torch.float32)
    else:
        # No edges => isolated nodes
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
    return X, y, edge_index, edge_weight, date_ranges


def train_gcn(
    train_panel: pd.DataFrame,
    val_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    train_snapshots: dict,
    val_snapshots: dict,
    test_snapshots: dict,
    feature_cols: list[str],
    stock_ids: list[str],
    cfg: dict,
) -> tuple[EvalResult, object]:
    """Train and evaluate the GCN with graph snapshots for each split."""
    gcn_cfg = cfg["training"]["gcn"]

    Xt, yt, ei_t, ew_t, _ = _snapshots_to_arrays(
        train_panel, train_snapshots, stock_ids, feature_cols
    )
    Xv, yv, ei_v, ew_v, _ = _snapshots_to_arrays(
        val_panel, val_snapshots, stock_ids, feature_cols
    )
    Xe, ye, ei_e, ew_e, _ = _snapshots_to_arrays(
        test_panel, test_snapshots, stock_ids, feature_cols
    )
    log.info("GCN inputs: train=%s val=%s test=%s", tuple(Xt.shape), tuple(Xv.shape), tuple(Xe.shape))

    model = fit_gcn(
        X=Xt, y=yt, edge_index=ei_t, edge_weight=ew_t,
        X_val=Xv, y_val=yv, ei_val=ei_v, ew_val=ew_v,
        **gcn_cfg, verbose=False,
    )
    p = predict_gcn(model, Xe, ei_e, ew_e)
    # Industry & regime lookups need external data; the caller wraps this
    # function with `_eval_with_meta` (see scripts/run_pipeline.py) before the
    # GCN result is finalised, so we return the raw probabilities here.
    from src.training.metrics import EvalResult
    res = EvalResult(name="GCN", n_samples=len(test_panel))
    return res, model, p


# -----------------------------------------------------------------------------#
# Helper: building snapshots aligned with a panel
# -----------------------------------------------------------------------------#
def build_split_snapshots(
    returns: pd.DataFrame,
    fund_holdings: pd.DataFrame,
    stock_industry: pd.Series,
    panel: pd.DataFrame,
    cfg: dict,
    *,
    freq: int = 20,
    lookback: int = 60,
) -> dict:
    """Build graph snapshots for every date in the (filtered) panel."""
    from src.graphs.snapshots import build_snapshot

    dates = select_graph_dates(
        panel.index.get_level_values("date").unique().sort_values(), freq=freq
    )
    snapshots: dict = {}
    for d in dates:
        try:
            snapshots[d] = build_snapshot(
                returns=returns,
                fund_holdings=fund_holdings,
                stock_industry=stock_industry,
                date=d,
                lookback=lookback,
                method="union",
                threshold_quantile=cfg["graph"]["threshold"]["quantiles"][-1],
                knn_k=cfg["graph"]["knn"]["k_values"][1],
                fund_overlap_quantile=cfg["graph"]["fund_holding"]["overlap_quantile"],
            )
        except Exception as e:  # pragma: no cover
            log.warning("Snapshot for %s failed: %s", d, e)
    # Forward-fill: missing dates reuse the latest snapshot
    last_snap = None
    ordered = panel.index.get_level_values("date").unique().sort_values()
    out = {}
    for d in ordered:
        if d in snapshots:
            last_snap = snapshots[d]
            out[d] = last_snap
        elif last_snap is not None:
            out[d] = last_snap
    return out


# -----------------------------------------------------------------------------#
# Ablation study
# -----------------------------------------------------------------------------#
def feature_ablation_study(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    *,
    industry_map: pd.Series | None = None,
    regime_series: pd.Series | None = None,
) -> dict[str, EvalResult]:
    """Run XGBoost with progressively smaller feature sets."""
    groups = partition_features(feature_cols)

    def flat(group_lists: list[list[str]]) -> list[str]:
        cols: list[str] = []
        for group in group_lists:
            cols.extend(group)
        return cols

    drop_specs = {
        "all": feature_cols,                                              # full
        "no_return_vol": [c for c in feature_cols if c not in groups.return_vol],
        "no_fundamental": [c for c in feature_cols if c not in groups.fundamental],
        "no_momentum": [c for c in feature_cols if c not in groups.momentum],
        "no_network_stat": [c for c in feature_cols if c not in groups.network_stat],
    }
    out: dict[str, EvalResult] = {}
    for name, cols in drop_specs.items():
        if not cols:
            continue
        xgb_cfg = cfg["training"]["baseline"]["xgb"]
        model = fit_xgboost(train[cols].values, train["y_dir"].values.astype(int), **xgb_cfg)
        p = model.predict_proba(test[cols].values)[:, 1]
        kwargs: dict = {"name": f"ablate:{name}"}
        if industry_map is not None:
            kwargs["industry_series"] = industry_map.reindex(
                test.index.get_level_values("stock_id")
            )
        if regime_series is not None:
            kwargs["regime_series"] = regime_series.reindex(
                test.index.get_level_values("date")
            )
        res = evaluate_predictions(test, p, **kwargs)
        out[name] = res
        log.info("Ablation %-22s AUC=%.4f Brier=%.4f n_feat=%d",
                 name, res.auc, res.brier, len(cols))
    return out


def gcn_edge_drop_curve(
    test_panel: pd.DataFrame,
    test_snapshots: dict,
    train_panel: pd.DataFrame,
    val_panel: pd.DataFrame,
    train_snapshots: dict,
    val_snapshots: dict,
    stock_ids: list[str],
    feature_cols: list[str],
    cfg: dict,
    *,
    drop_rates: list[float] | None = None,
) -> dict[float, float]:
    """Train once, then evaluate the same model against test graphs that have
    had progressively more edges removed."""
    if drop_rates is None:
        drop_rates = cfg["ablation"]["edge_drop_rates"]
    res_auc: dict[float, float] = {}
    # Re-use already trained model
    res, model, _ = train_gcn(
        train_panel, val_panel, test_panel,
        train_snapshots, val_snapshots, test_snapshots,
        feature_cols, stock_ids, cfg,
    )
    import numpy as np
    Xt, yt, ei_t, ew_t, _ = _snapshots_to_arrays(train_panel, train_snapshots, stock_ids, feature_cols)
    Xe, ye, ei_e, ew_e, _ = _snapshots_to_arrays(test_panel, test_snapshots, stock_ids, feature_cols)
    import torch
    rng = np.random.default_rng(cfg["data"]["seed"])
    for dr in drop_rates:
        ei2, ew2 = random_drop_edges(ei_e.numpy(), ew_e.numpy(), float(dr), rng=rng)
        ei2 = torch.tensor(ei2, dtype=torch.long)
        ew2 = torch.tensor(ew2, dtype=torch.float32)
        p = predict_gcn(model, Xe, ei2, ew2)
        from src.training.metrics import compute_auc
        auc = compute_auc(ye.numpy(), p)
        res_auc[float(dr)] = auc
        log.info("Edge-drop %.2f -> AUC=%.4f", dr, auc)
    return res_auc
