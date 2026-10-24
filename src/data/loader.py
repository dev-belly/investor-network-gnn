"""High-level data loader.

Orchestrates:
- Real data ingestion (parquet on disk) if present, otherwise
- Synthetic universe generation, persistence, and feature / label construction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.features import attach_labels, compute_node_features, make_labels
from src.data.synthetic import generate_universe
from src.utils.common import get_logger, set_seed

log = get_logger(__name__)


def load_or_generate(raw_dir: Path, cfg: dict, *, force: bool = False) -> LoadedData:
    """Return a :class:`LoadedData` bundle.

    If processed parquet files exist in ``raw_dir`` we re-use them; otherwise
    we generate a synthetic universe and persist everything.
    """
    raw_dir = Path(raw_dir)
    expected = [
        "prices.parquet",
        "returns.parquet",
        "fundamentals.parquet",
        "fund_holdings.parquet",
        "stock_industry.parquet",
        "market_regime.parquet",
    ]
    have_all = all((raw_dir / p).exists() for p in expected)

    if not have_all or force:
        log.info("Generating synthetic universe into %s", raw_dir)
        set_seed(cfg["data"]["seed"])
        bundle = generate_universe(
            n_stocks=cfg["data"]["n_stocks"],
            n_days=cfg["data"]["n_days"],
            n_industries=cfg["data"]["n_industries"],
            start_date=cfg["data"]["start_date"],
            seed=cfg["data"]["seed"],
        )
        bundle.to_parquet(raw_dir)
    else:
        log.info("Loading cached universe from %s", raw_dir)

    # ---- Reload --------------------------------------------------------------
    from src.data.synthetic import SyntheticDataBundle

    bundle = SyntheticDataBundle.from_parquet(raw_dir)

    log.info("Building node features...")
    features, feat_names = compute_node_features(bundle.returns, bundle.fundamentals, cfg)
    log.info("  -> %d rows, %d feature columns", len(features), len(feat_names))

    log.info("Building labels...")
    labels = make_labels(bundle.returns, cfg)
    log.info("  -> %d labelled rows", len(labels))

    df = attach_labels(features, labels)
    log.info("Joined panel: %d rows, %d cols", len(df), df.shape[1])

    return LoadedData(
        bundle=bundle,
        features=features,
        labels=labels,
        panel=df,
        feature_names=feat_names,
    )


class LoadedData:
    """Container bundling raw + processed data."""

    def __init__(
        self,
        bundle,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        panel: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        self.bundle = bundle
        self.features = features
        self.labels = labels
        self.panel = panel
        self.feature_names = feature_names

    # --- convenience --------------------------------------------------------
    def feature_matrix(
        self, split_dates: dict[str, pd.Timestamp] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return feature-only ``X`` and separate labels ``y`` in the same row order.

        Optionally restrict both frames to a date window. Future direction,
        risk and return labels must never appear among the model inputs.
        """
        cols = self.feature_names + ["y_dir", "y_risk", "fwd_ret"]
        df = self.panel[cols].copy()
        if split_dates is not None:
            start, end = split_dates.get("start"), split_dates.get("end")
            if start is not None:
                df = df[df.index.get_level_values("date") >= start]
            if end is not None:
                df = df[df.index.get_level_values("date") <= end]
        return df[self.feature_names].copy(), df[["y_dir", "y_risk", "fwd_ret"]].copy()

    def get_node_split(
        self, date: pd.Timestamp
    ) -> tuple[list[str], list[str]]:
        """Return stock_ids whose features exist at ``date`` (intersection of
        universe and the feature panel for that date)."""
        try:
            ids = self.features.xs(date, level="date").index.tolist()
        except KeyError:
            ids = self.bundle.returns.columns.tolist()
        return ids, self.bundle.stock_industry.loc[ids].tolist()
