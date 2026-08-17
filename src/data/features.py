"""Node feature engineering: returns, volatility, momentum, fundamentals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_multi_index(series: pd.Series, dates, stock_ids) -> pd.Series:
    """Re-attach (date, stock_id) MultiIndex to a series produced by ``stack()``.

    ``stack()`` turns a DataFrame into a Series with MultiIndex(date, col), so
    we relabel the column level to ``stock_id`` and ensure the names are
    exactly ``("date", "stock_id")``.
    """
    if not isinstance(series.index, pd.MultiIndex):
        # 1-D series (e.g. a single column): unlikely but handle it
        return series

    if series.index.names != ["date", "stock_id"]:
        new_names = list(series.index.names)
        # If it is (date, <something>), replace the second name
        if len(new_names) == 2:
            new_names[1] = "stock_id"
        series.index = series.index.set_names(new_names)
    return series


def _normalize_panel(panels: list[pd.Series | pd.DataFrame]) -> pd.DataFrame:
    """Join a list of stacked Series / DataFrames along ``(date, stock_id)``."""
    frames: list[pd.DataFrame] = []
    for p in panels:
        if isinstance(p, pd.Series):
            p = p.to_frame(p.name or "_")
        p = p.copy()
        if not isinstance(p.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex (date, stock_id)")
        if p.index.names != ["date", "stock_id"]:
            p.index = p.index.set_names(
                ["date" if n is None else n for n in p.index.names]
            )
            names = list(p.index.names)
            names[1] = "stock_id"
            p.index = p.index.set_names(names)
        frames.append(p)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out = out.sort_index()
    return out


def compute_node_features(
    returns: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a long-format node-feature frame.

    The frame is indexed by ``(date, stock_id)`` and contains one column per
    feature.  Missing values are forward-filled within each stock then dropped.
    """
    feat_cfg = cfg["features"]
    return_windows: list[int] = feat_cfg["return_windows"]
    vol_windows: list[int] = feat_cfg["volatility_windows"]
    mom_windows: list[int] = feat_cfg["momentum_windows"]

    panels: list[pd.Series | pd.DataFrame] = []

    # --- Cumulative returns / momentum signals -------------------------------
    for w in mom_windows:
        cum_ret = (1 + returns).rolling(window=w, min_periods=w // 2).apply(
            np.prod, raw=True
        ) - 1.0
        s = cum_ret.stack().rename(f"momentum_{w}d")
        s.index = s.index.set_names(["date", "stock_id"])
        panels.append(s)

    # --- Rolling mean returns -------------------------------------------------
    for w in return_windows:
        s = returns.rolling(window=w, min_periods=w // 2).mean().stack().rename(
            f"ret_mean_{w}d"
        )
        s.index = s.index.set_names(["date", "stock_id"])
        panels.append(s)

    # --- Rolling volatility (realized) ---------------------------------------
    for w in vol_windows:
        s = returns.rolling(window=w, min_periods=w // 2).std().stack().rename(
            f"vol_{w}d"
        )
        s.index = s.index.set_names(["date", "stock_id"])
        panels.append(s)

    # --- Idiosyncratic volatility proxy: abs return rolling mean ------------
    for w in (5, 20):
        s = returns.abs().rolling(window=w, min_periods=2).mean().stack().rename(
            f"abs_ret_{w}d"
        )
        s.index = s.index.set_names(["date", "stock_id"])
        panels.append(s)

    # --- Skewness & kurtosis (20d) -------------------------------------------
    s_skew = returns.rolling(20, min_periods=10).skew().stack().rename("skew_20d")
    s_skew.index = s_skew.index.set_names(["date", "stock_id"])
    s_kurt = returns.rolling(20, min_periods=10).kurt().stack().rename("kurt_20d")
    s_kurt.index = s_kurt.index.set_names(["date", "stock_id"])
    panels.extend([s_skew, s_kurt])

    # --- Fundamentals (static, broadcast across dates) -----------------------
    fund_cols = [
        "log_market_cap",
        "beta",
        "roe",
        "asset_growth",
        "leverage",
        "pe_ttm",
        "pb",
        "earnings_yield",
        "turnover",
    ]
    fund_block = []
    for d in returns.index:
        sub = fundamentals[fund_cols].copy()
        sub.index = pd.MultiIndex.from_product(
            [[d], sub.index], names=["date", "stock_id"]
        )
        fund_block.append(sub)
    fund_panel = pd.concat(fund_block, axis=0).sort_index()
    panels.append(fund_panel)

    # --- Concatenate everything ----------------------------------------------
    features = _normalize_panel(panels)
    features = features.replace([np.inf, -np.inf], np.nan)
    # Forward fill within each stock; then drop the warm-up rows
    features = (
        features.groupby(level="stock_id", group_keys=False)
        .apply(lambda g: g.ffill())
    )
    features = features.dropna()
    feature_names = list(features.columns)
    return features, feature_names


def make_labels(
    returns: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Construct prediction labels.

    Two labels are produced:

    - ``y_dir``  : binary, 1 if forward ``horizon``-day return is in the top
      half of the cross-section, 0 otherwise.
    - ``y_risk`` : ternary regime (low/mid/high) based on forward realized
      volatility quantiles.

    Output is indexed by ``(date, stock_id)``.
    """
    horizon = cfg["label"]["horizon"]
    risk_q = cfg["label"]["risk_quantiles"]

    fwd_ret = (
        (1 + returns)
        .rolling(window=horizon, min_periods=horizon)
        .apply(np.prod, raw=True)
        .shift(-horizon)
        - 1.0
    )
    fwd_vol = returns.rolling(window=horizon, min_periods=horizon // 2).std().shift(-horizon)

    cs_median = fwd_ret.median(axis=1)
    y_dir = (fwd_ret.gt(cs_median, axis=0)).astype(int)

    # Risk regime: tertile of forward vol (vectorized)
    q20 = fwd_vol.quantile(risk_q[0], axis=1)
    q80 = fwd_vol.quantile(risk_q[1], axis=1)
    y_risk = pd.DataFrame(
        np.where(fwd_vol.lt(q20, axis=0), 0,
                 np.where(fwd_vol.gt(q80, axis=0), 2, 1)),
        index=fwd_vol.index, columns=fwd_vol.columns,
    ).astype(int)

    fwd_ret_label = fwd_ret.stack().rename("fwd_ret")
    fwd_ret_label.index = fwd_ret_label.index.set_names(["date", "stock_id"])
    y_dir_label = y_dir.stack().rename("y_dir")
    y_dir_label.index = y_dir_label.index.set_names(["date", "stock_id"])
    y_risk_label = y_risk.stack().rename("y_risk")
    y_risk_label.index = y_risk_label.index.set_names(["date", "stock_id"])

    labels = pd.concat([fwd_ret_label, y_dir_label, y_risk_label], axis=1)
    labels.index.names = ["date", "stock_id"]
    labels = labels.dropna()
    return labels


def attach_labels(
    features: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Merge features and labels on the shared (date, stock_id) index."""
    df = features.join(labels, how="inner")
    df = df.dropna()
    return df
