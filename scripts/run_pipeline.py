"""End-to-end pipeline.

Usage::

    python scripts/run_pipeline.py                # full pipeline, synthetic data
    python scripts/run_pipeline.py --quick        # smoke-test mode
    python scripts/run_pipeline.py --skip-gcn      # baselines only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as `python scripts/run_pipeline.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_or_generate
from src.training import plots
from src.training.ablation import partition_features
from src.training.metrics import evaluate_predictions
from src.training.split import TimeSplit, chronological_split, filter_by_split
from src.training.trainer import (
    build_split_snapshots,
    feature_ablation_study,
    gcn_edge_drop_curve,
    train_classical_baselines,
    train_gcn,
)
from src.utils.common import (
    Paths,
    get_logger,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

log = get_logger("pipeline")


def _format_metric(v) -> str:
    return f"{v:.4f}" if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "n/a"


def build_report(eval_results, ablations, edge_drop, regime_rows, n_features_used) -> str:
    lines: list[str] = []
    lines.append("# Investor Network GCN — Experiment Report\n")
    lines.append("## 1. Test-set metrics\n")
    lines.append("| Model | n | AUC | Brier | ECE | Top-10 hit | Long-Short |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in eval_results:
        lines.append(
            f"| {r.name} | {r.n_samples} | {_format_metric(r.auc)} | "
            f"{_format_metric(r.brier)} | {_format_metric(r.ece)} | "
            f"{_format_metric(r.top_hit)} | {_format_metric(r.long_short)} |"
        )

    lines.append("\n## 2. Feature ablation (XGBoost)\n")
    lines.append("| Setting | n_feat | AUC | Brier |")
    lines.append("|---|---|---|---|")
    for name, (cols, res) in ablations.items():
        lines.append(
            f"| {name} | {len(cols)} | {_format_metric(res.auc)} | "
            f"{_format_metric(res.brier)} |"
        )

    lines.append("\n## 3. Edge-drop ablation (GCN)\n")
    lines.append("| drop_rate | AUC |")
    lines.append("|---|---|")
    for dr, auc in sorted(edge_drop.items()):
        lines.append(f"| {dr:.2f} | {_format_metric(auc)} |")

    lines.append("\n## 4. AUC by market regime\n")
    lines.append("| Model | bull | bear | sideways |")
    lines.append("|---|---|---|---|")
    for row in regime_rows:
        lines.append(
            f"| {row['model']} | {_format_metric(row.get('bull'))} | "
            f"{_format_metric(row.get('bear'))} | {_format_metric(row.get('sideways'))} |"
        )

    lines.append("\n## 5. Per-industry highlights\n")
    if eval_results:
        gcn_res = next((r for r in eval_results if "GCN" in r.name), eval_results[0])
        items = sorted(gcn_res.per_industry.items(), key=lambda x: -(x[1].get("auc") or 0))[:5]
        worst = sorted(gcn_res.per_industry.items(), key=lambda x: (x[1].get("auc") or 0))[:5]
        lines.append("Top industries by GCN AUC:")
        for name, vals in items:
            lines.append(f"- {name}: AUC={_format_metric(vals.get('auc'))}, n={vals.get('n')}")
        lines.append("\nWorst industries by GCN AUC:")
        for name, vals in worst:
            lines.append(f"- {name}: AUC={_format_metric(vals.get('auc'))}, n={vals.get('n')}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the investor-network GCN pipeline.")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    p.add_argument("--quick", action="store_true", help="Smoke test with small universe and few epochs.")
    p.add_argument("--medium", action="store_true", help="Medium run (~5-10 min, 200 stocks x 1000 days).")
    p.add_argument("--skip-gcn", action="store_true", help="Skip the GCN model (baselines only).")
    p.add_argument("--no-ablation", action="store_true", help="Skip ablation studies.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = load_config(args.config)

    if args.quick:
        cfg["data"]["n_stocks"] = 80
        cfg["data"]["n_days"] = 600
        cfg["training"]["gcn"]["epochs"] = 20
        cfg["training"]["baseline"]["mlp"]["epochs"] = 10
    elif args.medium:
        cfg["data"]["n_stocks"] = 200
        cfg["data"]["n_days"] = 1000
        cfg["training"]["gcn"]["epochs"] = 60
        cfg["training"]["baseline"]["mlp"]["epochs"] = 40

    set_seed(cfg["data"]["seed"])
    paths = Paths.from_root(PROJECT_ROOT).ensure()
    log.info("Pipeline starting (project_root=%s, quick=%s)", PROJECT_ROOT, args.quick)

    # ------------------------------------------------------------------ #
    # 1) Load (or generate) the universe
    # ------------------------------------------------------------------ #
    log.info("[1/8] Loading universe ...")
    data = load_or_generate(paths.data_raw, cfg)
    bundle = data.bundle
    panel = data.panel
    feature_names = data.feature_names
    log.info("  panel shape=%s  feature_dim=%d", panel.shape, len(feature_names))

    # ------------------------------------------------------------------ #
    # 2) Time split
    # ------------------------------------------------------------------ #
    log.info("[2/8] Building time split ...")
    dates = panel.index.get_level_values("date").unique().sort_values()
    split = chronological_split(
        dates,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        purge_gap=cfg["split"]["purge_gap"],
    )
    log.info("  train=%s..%s  val=%s..%s  test=%s..%s",
             split.train[0].date(), split.train[1].date(),
             split.val[0].date(), split.val[1].date(),
             split.test[0].date(), split.test[1].date())

    train = filter_by_split(panel, split, "train")
    val = filter_by_split(panel, split, "val")
    test = filter_by_split(panel, split, "test")
    log.info("  train rows=%d  val rows=%d  test rows=%d", len(train), len(val), len(test))

    # Stock universe (use the test universe as canonical)
    stock_ids = bundle.returns.columns.tolist()
    stock_industry_map = bundle.stock_industry

    # ------------------------------------------------------------------ #
    # 3) Save data info
    # ------------------------------------------------------------------ #
    save_json(
        {
            "panel_shape": list(panel.shape),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "split": split.as_dict(),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "universe_size": len(stock_ids),
            "n_industries": stock_industry_map.nunique(),
        },
        paths.reports / "data_summary.json",
    )

    # ------------------------------------------------------------------ #
    # 4) Train classical baselines
    # ------------------------------------------------------------------ #
    log.info("[3/8] Training classical baselines (LR / XGBoost / MLP) ...")
    industry_series = stock_industry_map.reindex(panel.index.get_level_values("stock_id"))
    regime_series = bundle.market_regime.reindex(panel.index.get_level_values("date"))

    # Helper for evaluate_predictions with proper industry mapping
    def _industry_for(df):
        return stock_industry_map.reindex(df.index.get_level_values("stock_id"))

    # We need to wrap evaluate_predictions because the trainer version already imports it.
    # Simpler: re-import and pass correctly.
    from src.training.metrics import evaluate_predictions as _eval

    def eval_with_meta(df, probs, name):
        return _eval(
            df, probs, name=name,
            industry_series=_industry_for(df),
            regime_series=bundle.market_regime.reindex(df.index.get_level_values("date")),
        )

    # Manually replicate trainer.train_classical_baselines but use the wrapped eval
    from sklearn.preprocessing import StandardScaler
    from src.models.baselines import (
        fit_logreg,
        fit_mlp,
        fit_xgboost,
        predict_mlp,
    )
    Xt = train[feature_names].values
    yt = train["y_dir"].values.astype(int)
    Xv = val[feature_names].values
    yv = val["y_dir"].values.astype(int)
    Xe = test[feature_names].values
    ye = test["y_dir"].values.astype(int)

    eval_results: list = []
    models: dict = {}

    lr_model = fit_logreg(Xt, yt)
    p_lr = lr_model.predict_proba(Xe)[:, 1]
    models["logreg"] = lr_model
    eval_results.append(eval_with_meta(test, p_lr, "LogReg"))

    xgb_cfg = cfg["training"]["baseline"]["xgb"]
    xgb = fit_xgboost(Xt, yt, **xgb_cfg)
    p_xgb = xgb.predict_proba(Xe)[:, 1]
    models["xgboost"] = xgb
    eval_results.append(eval_with_meta(test, p_xgb, "XGBoost"))

    mlp_cfg = cfg["training"]["baseline"]["mlp"]
    mlp = fit_mlp(Xt, yt, Xv, yv, **mlp_cfg)
    p_mlp = predict_mlp(mlp, Xe)
    models["mlp"] = mlp
    eval_results.append(eval_with_meta(test, p_mlp, "MLP-no-graph"))

    # ------------------------------------------------------------------ #
    # 5) GCN snapshots & GCN training
    # ------------------------------------------------------------------ #
    gcn_res = None
    edge_drop_curve = {}
    if not args.skip_gcn:
        log.info("[4/8] Building graph snapshots ...")
        t0 = time.time()
        train_snaps = build_split_snapshots(
            bundle.returns, bundle.fund_holdings, bundle.stock_industry, train, cfg, freq=20
        )
        val_snaps = build_split_snapshots(
            bundle.returns, bundle.fund_holdings, bundle.stock_industry, val, cfg, freq=20
        )
        test_snaps = build_split_snapshots(
            bundle.returns, bundle.fund_holdings, bundle.stock_industry, test, cfg, freq=20
        )
        log.info("  snapshots: train=%d val=%d test=%d (%.1fs)",
                 len(train_snaps), len(val_snaps), len(test_snaps), time.time() - t0)

        log.info("[5/8] Training GCN ...")
        gcn_res, gcn_model, p_gcn = train_gcn(
            train, val, test, train_snaps, val_snaps, test_snaps,
            feature_names, stock_ids, cfg,
        )
        # Wrap with industry & regime metadata
        gcn_res = _eval(test, p_gcn, name="GCN",
                        industry_series=_industry_for(test),
                        regime_series=bundle.market_regime.reindex(
                            test.index.get_level_values("date")))
        eval_results.append(gcn_res)
        models["gcn"] = gcn_model

        # Edge-drop ablation
        if not args.no_ablation:
            log.info("[6/8] Edge-drop ablation ...")
            edge_drop_curve = gcn_edge_drop_curve(
                test, test_snaps, train, val,
                train_snaps, val_snaps, stock_ids, feature_names, cfg,
            )
    else:
        log.info("[4-6/8] Skipping GCN per --skip-gcn")

    # ------------------------------------------------------------------ #
    # 6) Feature ablation
    # ------------------------------------------------------------------ #
    ablations: dict = {}
    if not args.no_ablation:
        log.info("[7/8] Feature-group ablation (XGBoost) ...")
        ablations = feature_ablation_study(
            train, val, test, feature_names, cfg,
            industry_map=stock_industry_map,
            regime_series=bundle.market_regime,
        )
        # Wrap each (cols, res) so we can report n_feat used
        groups = partition_features(feature_names)
        renamed: dict = {}
        for old_name, res in ablations.items():
            if old_name == "no_return_vol":
                cols = [c for c in feature_names if c not in groups.return_vol]
            elif old_name == "no_fundamental":
                cols = [c for c in feature_names if c not in groups.fundamental]
            elif old_name == "no_momentum":
                cols = [c for c in feature_names if c not in groups.momentum]
            elif old_name == "no_network_stat":
                cols = [c for c in feature_names if c not in groups.network_stat]
            else:
                cols = feature_names
            renamed[old_name] = (cols, res)
        ablations = renamed

    # ------------------------------------------------------------------ #
    # 7) Visualisations
    # ------------------------------------------------------------------ #
    log.info("[8/8] Rendering plots ...")
    # --- Correlation distribution
    sample_window = bundle.returns.iloc[-252:]
    corr = sample_window.corr().values
    plots.plot_corr_distribution(corr, paths.figures / "01_corr_distribution.png")

    # --- A graph snapshot
    from src.graphs.snapshots import build_snapshot

    snap = build_snapshot(
        returns=bundle.returns,
        fund_holdings=bundle.fund_holdings,
        stock_industry=bundle.stock_industry,
        date=test.index.get_level_values("date")[-1],
        lookback=60,
        method="union",
        threshold_quantile=cfg["graph"]["threshold"]["quantiles"][-1],
        knn_k=cfg["graph"]["knn"]["k_values"][1],
        fund_overlap_quantile=cfg["graph"]["fund_holding"]["overlap_quantile"],
    )
    import networkx as nx

    g_vis = nx.Graph()
    g_vis.add_nodes_from(range(len(stock_ids)))
    if snap.edge_index.size:
        for (u, v), w in zip(snap.edge_index.T, snap.edge_weight):
            if u == v:
                continue
            g_vis.add_edge(int(u), int(v), weight=float(w))
    # Subsample for visualisation
    if g_vis.number_of_edges() > 1500:
        rng = np.random.default_rng(0)
        keep = rng.choice(g_vis.number_of_edges(), 1500, replace=False)
        edges_to_remove = [list(g_vis.edges())[i] for i in range(g_vis.number_of_edges()) if i not in set(keep)]
        g_vis.remove_edges_from(edges_to_remove)
    plots.plot_graph(g_vis, paths.figures / "02_graph_snapshot.png", title="Union graph snapshot")

    # --- ROC curves
    probs_dict = {"LogReg": p_lr, "XGBoost": p_xgb, "MLP-no-graph": p_mlp}
    if gcn_res is not None:
        probs_dict["GCN"] = p_gcn

    roc_inputs = {name: (test["y_dir"].values, p) for name, p in probs_dict.items()}
    plots.plot_roc_curves(roc_inputs, paths.figures / "03_roc_curves.png")
    plots.plot_calibration(test, probs_dict, paths.figures / "04_calibration.png")
    plots.plot_metric_bar(eval_results, paths.figures / "05_metrics_bar.png", metric="auc")
    plots.plot_industry_auc(eval_results, paths.figures / "06_per_industry_auc.png")
    plots.plot_regime_auc(eval_results, paths.figures / "07_per_regime_auc.png")

    if "GCN" in probs_dict:
        plots.plot_long_short_curve(test, probs_dict["GCN"], paths.figures / "08_gcn_long_short.png")
    if edge_drop_curve:
        plots.plot_ablation_curve(edge_drop_curve, paths.figures / "09_edge_drop_curve.png")

    # --- Feature importance (XGBoost)
    plots.plot_feature_importance(models["xgboost"], feature_names,
                                  paths.figures / "10_xgb_feature_importance.png")

    # ------------------------------------------------------------------ #
    # 8) Save JSON + Markdown report
    # ------------------------------------------------------------------ #
    log.info("Saving reports ...")
    summary = {
        "models": [r.to_dict() for r in eval_results],
        "ablations": {k: {"n_feat": len(v[0]), "metrics": v[1].to_dict()} for k, v in ablations.items()},
        "edge_drop_ablation": {str(k): v for k, v in edge_drop_curve.items()},
        "config": {
            "data": cfg["data"],
            "graph": cfg["graph"],
            "split": cfg["split"],
            "training": cfg["training"],
        },
        "panel_shape": list(panel.shape),
    }
    save_json(summary, paths.reports / "results.json")

    # regime pivot
    regime_rows = []
    for r in eval_results:
        row = {"model": r.name}
        for reg, vals in r.per_regime.items():
            row[reg] = vals.get("auc")
        regime_rows.append(row)

    md = build_report(eval_results, ablations, edge_drop_curve, regime_rows, len(feature_names))
    plots.save_text_report(md, paths.reports / "results.md")

    log.info("Done.  Outputs are under %s.", paths.root / "outputs")
    return 0


class EvalResultShim:
    """Tiny adapter so ablation rows carry their feature-column count."""
    def __init__(self, cols: list[str], res):
        self.cols = cols
        self.auc = res.auc
        self.brier = res.brier
        self.ece = res.ece
        self.top_hit = res.top_hit
        self.long_short = res.long_short
        self.n_samples = res.n_samples
        self.name = res.name
        self.per_industry = res.per_industry
        self.per_regime = res.per_regime

    def as_dict(self):
        d = self.__dict__.copy()
        return d


if __name__ == "__main__":
    sys.exit(main())
