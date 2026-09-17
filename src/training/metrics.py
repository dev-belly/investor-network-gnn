"""Evaluation metrics for binary direction prediction.

We focus on standard classification metrics that are robust across markets:

- **AUC** (ROC-AUC) — ranking quality under class balance.
- **Brier score** — calibration (lower is better; probabilistic squared loss).
- **Expected Calibration Error (ECE)** — bin-based reliability.
- **Top-decile hit rate** — among the highest-prob stocks, how many actually
  realised a positive return (i.e. precision-at-10%).
- **Information / decile backtest** — long top-decile vs short bottom-decile.

We also compute **per-industry** and **per-regime** slices (bull/bear/sideways).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


# -----------------------------------------------------------------------------#
# Metric helpers
# -----------------------------------------------------------------------------#
def compute_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def compute_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y_true, p))


def compute_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = ((p >= lo) if lo == 0 else (p > lo)) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def top_decile_hit_rate(
    dates: pd.Series, y: np.ndarray, p: np.ndarray, fwd_ret: np.ndarray, *, q: float = 0.1
) -> float:
    """For each date, take the top-q predicted stocks and check forward return > 0."""
    df = pd.DataFrame({"d": dates.values, "y": y, "p": p, "r": fwd_ret})
    hits = []
    for _d, sub in df.groupby("d"):
        if len(sub) < 20:
            continue
        thr = sub["p"].quantile(1 - q)
        top = sub[sub["p"] >= thr]
        if len(top) == 0:
            continue
        hits.append((top["r"] > 0).mean())
    return float(np.nanmean(hits)) if hits else float("nan")


def long_short_decile_return(
    dates: pd.Series, p: np.ndarray, fwd_ret: np.ndarray, *, q: float = 0.1
) -> float:
    """Long top-q, short bottom-q, equal-weighted.  Average return over time."""
    df = pd.DataFrame({"d": dates.values, "p": p, "r": fwd_ret})
    rets = []
    for _d, sub in df.groupby("d"):
        if len(sub) < 20:
            continue
        long_thr = sub["p"].quantile(1 - q)
        short_thr = sub["p"].quantile(q)
        long_ret = sub[sub["p"] >= long_thr]["r"].mean()
        short_ret = sub[sub["p"] <= short_thr]["r"].mean()
        if not np.isnan(long_ret) and not np.isnan(short_ret):
            rets.append(long_ret - short_ret)
    return float(np.nanmean(rets)) if rets else float("nan")


# -----------------------------------------------------------------------------#
# Evaluation context
# -----------------------------------------------------------------------------#
@dataclass
class EvalResult:
    name: str
    n_samples: int
    auc: float = float("nan")
    brier: float = float("nan")
    ece: float = float("nan")
    top_hit: float = float("nan")
    long_short: float = float("nan")
    per_industry: dict[str, dict] = field(default_factory=dict)
    per_regime: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "auc": self.auc,
            "brier": self.brier,
            "ece": self.ece,
            "top_hit": self.top_hit,
            "long_short": self.long_short,
            "per_industry": self.per_industry,
            "per_regime": self.per_regime,
        }


def evaluate_predictions(
    panel: pd.DataFrame,
    probs: np.ndarray,
    *,
    name: str,
    industry_series: pd.Series | None = None,
    regime_series: pd.Series | None = None,
) -> EvalResult:
    """Run the canonical metric suite and slice by industry / regime.

    Parameters
    ----------
    panel : DataFrame
        Indexed ``(date, stock_id)``; must contain ``y_dir`` and ``fwd_ret``.
    probs : array
        Model probabilities aligned with ``panel`` rows.
    """
    panel = panel.copy()
    panel["p"] = probs
    panel["y_dir"] = panel["y_dir"].astype(int)
    panel["fwd_ret"] = panel["fwd_ret"].astype(float)
    y = panel["y_dir"].values
    p = panel["p"].values
    fwd = panel["fwd_ret"].values
    dates = panel.index.get_level_values("date")

    overall = EvalResult(
        name=name,
        n_samples=len(panel),
        auc=compute_auc(y, p),
        brier=compute_brier(y, p),
        ece=compute_ece(y, p),
        top_hit=top_decile_hit_rate(dates, y, p, fwd),
        long_short=long_short_decile_return(dates, p, fwd),
    )

    # Per-industry slice
    if industry_series is not None:
        ind = industry_series.reindex(panel.index.get_level_values("stock_id")).values
        for ind_name in pd.unique(ind):
            mask = ind == ind_name
            if mask.sum() < 50:
                continue
            sub = panel[mask]
            sub_dates = sub.index.get_level_values("date")
            overall.per_industry[str(ind_name)] = {
                "n": int(mask.sum()),
                "auc": compute_auc(sub["y_dir"].values, sub["p"].values),
                "long_short": long_short_decile_return(
                    sub_dates, sub["p"].values, sub["fwd_ret"].values
                ),
            }

    # Per-regime slice
    if regime_series is not None:
        regime = regime_series.reindex(panel.index.get_level_values("date")).values
        for reg in pd.unique(regime):
            mask = regime == reg
            if mask.sum() < 50:
                continue
            sub = panel[mask]
            sub_dates = sub.index.get_level_values("date")
            overall.per_regime[str(reg)] = {
                "n": int(mask.sum()),
                "auc": compute_auc(sub["y_dir"].values, sub["p"].values),
                "long_short": long_short_decile_return(
                    sub_dates, sub["p"].values, sub["fwd_ret"].values
                ),
            }

    return overall
