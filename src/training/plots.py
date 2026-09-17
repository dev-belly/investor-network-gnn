"""Visualization helpers for the pipeline.

All functions return a matplotlib figure and write to ``paths.figures`` as PNG.
We deliberately avoid heavy plotting libraries (plotly, seaborn objects) so
the project runs on a stock pip install.
"""
from __future__ import annotations

import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import get_logger

log = get_logger(__name__)

# Sane defaults for headless rendering
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ---- Font selection ---------------------------------------------------------
# Resolve a CJK-capable font so Chinese industry labels render correctly.
def _set_cjk_font() -> None:
    candidates = []
    if platform.system() == "Darwin":
        candidates = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Songti.ttc",
        ]
    elif platform.system() == "Linux":
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]
    elif platform.system() == "Windows":
        candidates = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for path in candidates:
        if Path(path).exists():
            try:
                from matplotlib import font_manager

                font_manager.fontManager.addfont(path)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.sans-serif"] = [plt.rcParams["font.family"]] + plt.rcParams.get(
                    "font.sans-serif", []
                )
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue


_set_cjk_font()


# -----------------------------------------------------------------------------#
# Figures
# -----------------------------------------------------------------------------#
def plot_corr_distribution(corr_matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    ax.hist(off_diag, bins=60, color="#5B7FFF", alpha=0.8, edgecolor="white")
    ax.axvline(np.quantile(off_diag, 0.85), ls="--", color="red", label="85th pct")
    ax.set_title("Pairwise return correlation distribution")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_graph(graph, path: Path, *, title: str = "Graph") -> None:
    import networkx as nx

    pos = nx.spring_layout(graph, seed=0, k=0.4)
    fig, ax = plt.subplots(figsize=(7, 6))
    if graph.number_of_edges() > 0:
        edges_data = list(graph.edges(data=True))
        if len(edges_data[0]) == 3:
            _, _, data_list = zip(*edges_data)
            weights = np.array([d.get("weight", 1.0) for d in data_list])
        else:
            weights = np.ones(graph.number_of_edges())
    else:
        weights = np.array([])
    if weights.size:
        max_w = weights.max()
        widths = 0.2 + 2.0 * (weights / max(max_w, 1e-6))
    else:
        widths = np.array([])
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=30, alpha=0.7, node_color="#444")
    if widths.size:
        nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, alpha=0.35, edge_color="#999")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc_curves(results: dict, path: Path) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in results.items():
        if np.isnan(p).any():
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name, lw=1.5)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(panel: pd.DataFrame, probs_dict: dict[str, np.ndarray], path: Path) -> None:
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs_dict.items():
        y = panel["y_dir"].values
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=name, lw=1.5)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_bar(eval_results: list, path: Path, *, metric: str = "auc", title: str | None = None) -> None:
    names = [r.name for r in eval_results]
    values = [getattr(r, metric) for r in eval_results]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color="#5B7FFF", edgecolor="white")
    ax.set_title(title or f"{metric.upper()} by model")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, max(1.0, max(values) * 1.1 if values else 1.0))
    for b, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ablation_curve(curve: dict, path: Path, *, xlabel: str = "drop rate", ylabel: str = "AUC") -> None:
    xs = list(curve.keys())
    ys = list(curve.values())
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", color="#5B7FFF")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("GCN edge-drop ablation")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(model, feature_names: list[str], path: Path, *, top_k: int = 20) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[order], importances[order], color="#5B7FFF")
    ax.set_title("XGBoost top feature importances")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_industry_auc(eval_results: list, path: Path) -> None:
    rows = []
    for r in eval_results:
        for ind, vals in r.per_industry.items():
            rows.append({"model": r.name, "industry": ind, "auc": vals.get("auc", np.nan)})
    if not rows:
        return
    df = pd.DataFrame(rows).dropna()
    pivot = df.pivot(index="industry", columns="model", values="auc")
    pivot = pivot.sort_values(pivot.columns[0], ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(pivot))))
    pivot.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel("AUC")
    ax.set_title("Per-industry AUC by model")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_regime_auc(eval_results: list, path: Path) -> None:
    rows = []
    for r in eval_results:
        for reg, vals in r.per_regime.items():
            rows.append({"model": r.name, "regime": reg, "auc": vals.get("auc", np.nan)})
    if not rows:
        return
    df = pd.DataFrame(rows).dropna()
    pivot = df.pivot(index="regime", columns="model", values="auc")
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("AUC")
    ax.set_title("AUC by market regime (bull / bear / sideways)")
    ax.set_ylim(0.45, 0.75)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_long_short_curve(panel: pd.DataFrame, probs: np.ndarray, path: Path) -> None:
    df = panel.copy()
    df["p"] = probs
    df = df.reset_index()
    daily = []
    for d, sub in df.groupby("date"):
        if len(sub) < 30:
            continue
        top = sub["p"].quantile(0.9)
        bot = sub["p"].quantile(0.1)
        long_ret = sub[sub["p"] >= top]["fwd_ret"].mean()
        short_ret = sub[sub["p"] <= bot]["fwd_ret"].mean()
        daily.append({"date": d, "ls": long_ret - short_ret})
    s = pd.DataFrame(daily).set_index("date").sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s.index, s["ls"].values, color="#5B7FFF", lw=1.5)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_title("Forward-horizon top-minus-bottom decile spread")
    ax.set_ylabel("Overlapping horizon return; not portfolio NAV")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_text_report(report: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
