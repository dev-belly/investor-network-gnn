"""Synthetic A-share market data generator.

Why synthetic?
--------------
The project's value is in the **methodology** (graph construction, GNN
baselines, ablation, time-series evaluation).  Forcing the reader to chase a
Tushare / Wind token would obscure the science.  We therefore ship a
deterministic generator that reproduces the structural properties a quant
expects to see in HS300 data:

- 27 SW industries with intra-industry correlation higher than inter-industry
- A latent factor model (market / size / value / momentum) with realistic loadings
- Heavy-tailed daily returns
- Synthetic but mutually consistent fundamentals (ROE, growth, leverage, ...)
- Fund holdings that cluster around large caps and within industries

If real A-share data is available, drop the corresponding parquet files into
``data/raw/`` matching the schema produced here and ``loader.py`` will use
them instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------#
# Industry universe (simplified SW classification)
# -----------------------------------------------------------------------------#
INDUSTRY_NAMES = [
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "家用电器",
    "食品饮料", "纺织服饰", "轻工制造", "医药生物", "公用事业",
    "交通运输", "房地产", "商贸零售", "社会服务", "综合", "建筑材料",
    "建筑装饰", "电力设备", "机械设备", "汽车", "美容护理", "银行",
    "非银金融", "计算机", "传媒", "通信",
]


@dataclass
class SyntheticDataBundle:
    """Container for the generated universe."""

    prices: pd.DataFrame              # index: trading day, cols: stock_id
    returns: pd.DataFrame             # daily simple returns
    fundamentals: pd.DataFrame        # static per stock
    fund_holdings: pd.DataFrame       # long format: fund_id, stock_id, weight
    stock_industry: pd.Series         # stock_id -> industry
    market_regime: pd.Series          # date -> {bull, bear, sideways}
    trading_days: pd.DatetimeIndex

    def to_parquet(self, root: Path) -> None:
        """Persist every frame to ``root`` as parquet."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.prices.to_parquet(root / "prices.parquet")
        self.returns.to_parquet(root / "returns.parquet")
        self.fundamentals.to_parquet(root / "fundamentals.parquet")
        self.fund_holdings.to_parquet(root / "fund_holdings.parquet")
        self.stock_industry.to_frame("industry").to_parquet(root / "stock_industry.parquet")
        self.market_regime.to_frame("regime").to_parquet(root / "market_regime.parquet")

    @classmethod
    def from_parquet(cls, root: Path) -> "SyntheticDataBundle":
        root = Path(root)
        prices = pd.read_parquet(root / "prices.parquet")
        returns = pd.read_parquet(root / "returns.parquet")
        fundamentals = pd.read_parquet(root / "fundamentals.parquet")
        fund_holdings = pd.read_parquet(root / "fund_holdings.parquet")
        stock_industry = pd.read_parquet(root / "stock_industry.parquet")["industry"]
        market_regime = pd.read_parquet(root / "market_regime.parquet")["regime"]
        return cls(
            prices=prices,
            returns=returns,
            fundamentals=fundamentals,
            fund_holdings=fund_holdings,
            stock_industry=stock_industry,
            market_regime=market_regime,
            trading_days=prices.index,
        )


# -----------------------------------------------------------------------------#
# Generator
# -----------------------------------------------------------------------------#
def generate_universe(
    n_stocks: int = 300,
    n_days: int = 1500,
    n_industries: int = 27,
    n_funds: int = 120,
    start_date: str = "2018-01-01",
    seed: int = 42,
) -> SyntheticDataBundle:
    """Generate a realistic synthetic HS300-style universe.

    Parameters
    ----------
    n_stocks : int
        Number of stocks (default 300 for HS300).
    n_days : int
        Number of trading days (~6 years).
    n_industries : int
        Number of SW industries (we use 27, matching the latest SW taxonomy).
    n_funds : int
        Number of mutual funds to simulate.
    start_date : str
        First trading day (weekends are dropped below).
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # ---- Trading calendar (business days, drop weekends) ---------------------
    start = pd.Timestamp(start_date)
    trading_days = pd.bdate_range(start, periods=n_days)
    n_days_actual = len(trading_days)

    # ---- Stock metadata ------------------------------------------------------
    industries = INDUSTRY_NAMES[:n_industries]
    industry_assign = rng.choice(industries, size=n_stocks, replace=True)
    stock_ids = [f"S{str(i).zfill(4)}" for i in range(n_stocks)]
    stock_industry = pd.Series(industry_assign, index=stock_ids, name="industry")

    # Market cap proxy: lognormal, with concentration toward large caps
    log_cap = rng.normal(loc=10.0, scale=1.0, size=n_stocks)
    log_cap -= log_cap.min()  # ensure positivity
    market_cap = np.exp(log_cap)
    market_cap = pd.Series(market_cap, index=stock_ids, name="log_market_cap")

    # Beta to the market: 0.6 ~ 1.6 range
    beta = rng.normal(loc=1.0, scale=0.25, size=n_stocks)
    beta = np.clip(beta, 0.3, 2.0)
    beta = pd.Series(beta, index=stock_ids, name="beta")

    # Industry index for each stock (used for correlated shocks)
    industry_codes = pd.Categorical(industry_assign).codes
    n_ind = len(industries)

    # ---- Latent factor returns (market + 4 style factors) --------------------
    # 4 style factors: size, value, momentum, low-vol
    factor_returns = {
        "market": rng.normal(0.0004, 0.012, size=n_days_actual),
        "size": rng.normal(0.0, 0.006, size=n_days_actual),
        "value": rng.normal(0.0, 0.005, size=n_days_actual),
        "momentum": rng.normal(0.0, 0.008, size=n_days_actual),
        "lowvol": rng.normal(0.0, 0.005, size=n_days_actual),
    }
    factor_df = pd.DataFrame(factor_returns, index=trading_days)

    # ---- Industry factor returns (sector-specific shocks) --------------------
    # Each industry has a slow-moving factor, correlation with market ~ 0.4
    industry_factor = rng.normal(0.0, 0.006, size=(n_days_actual, n_ind))
    for i in range(n_ind):
        industry_factor[:, i] = (
            0.4 * factor_df["market"].values
            + np.sqrt(0.84) * industry_factor[:, i]
        )
    industry_factor_df = pd.DataFrame(
        industry_factor, index=trading_days, columns=industries
    )

    # ---- Idiosyncratic returns -----------------------------------------------
    # Heavy-tailed: mixture of normal and a Cauchy-like component
    def sample_idio(n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_t(df=5, size=n) * 0.008  # fatter tails
        return z

    # ---- Stock loadings on factors ------------------------------------------
    size_loading = -0.5 * (market_cap.values - market_cap.mean()) / market_cap.std()
    value_loading = rng.normal(0.0, 0.3, size=n_stocks)
    momentum_loading = rng.normal(0.0, 0.4, size=n_stocks)
    lowvol_loading = rng.normal(0.0, 0.3, size=n_stocks)

    # ---- Build stock return matrix ------------------------------------------
    returns_mat = np.zeros((n_days_actual, n_stocks))
    for t in range(n_days_actual):
        mkt_t = factor_df["market"].iloc[t]
        size_t = factor_df["size"].iloc[t]
        value_t = factor_df["value"].iloc[t]
        mom_t = factor_df["momentum"].iloc[t]
        low_t = factor_df["lowvol"].iloc[t]
        ind_fac_t = industry_factor_df.iloc[t].values  # (n_ind,)
        idio = sample_idio(n_stocks, rng)

        returns_mat[t] = (
            beta.values * mkt_t
            + size_loading * size_t
            + value_loading * value_t
            + momentum_loading * mom_t
            + lowvol_loading * low_t
            + ind_fac_t[industry_codes]
            + idio
        )

    # ---- Realized volatility regime (clustered) ------------------------------
    # GARCH(1,1)-like: a slow random walk for vol-of-vol
    vol_of_vol = np.zeros(n_days_actual)
    vol_of_vol[0] = 1.0
    for t in range(1, n_days_actual):
        vol_of_vol[t] = 0.97 * vol_of_vol[t - 1] + 0.03 * abs(rng.normal(0, 0.3))
    vol_of_vol = 0.7 + vol_of_vol  # ensure positive
    returns_mat = returns_mat * vol_of_vol[:, None]

    returns_df = pd.DataFrame(returns_mat, index=trading_days, columns=stock_ids)

    # ---- Build price series --------------------------------------------------
    prices = (1 + returns_df).cumprod() * 10.0  # arbitrary starting price
    prices.iloc[0] = 10.0

    # ---- Fundamentals (static, per stock) -----------------------------------
    fundamentals = pd.DataFrame(
        {
            "log_market_cap": market_cap,
            "beta": beta,
            "roe": rng.normal(0.10, 0.06, n_stocks).clip(-0.1, 0.4),
            "asset_growth": rng.normal(0.08, 0.10, n_stocks).clip(-0.2, 0.4),
            "leverage": rng.normal(0.45, 0.15, n_stocks).clip(0.1, 0.9),
            "pe_ttm": np.exp(rng.normal(2.5, 0.6, n_stocks)).clip(2, 200),
            "pb": rng.lognormal(0.5, 0.5, n_stocks).clip(0.1, 20),
            "earnings_yield": rng.normal(0.05, 0.03, n_stocks).clip(-0.1, 0.25),
            "turnover": rng.lognormal(0.0, 0.5, n_stocks).clip(0.001, 5),
        },
        index=stock_ids,
    )

    # ---- Fund holdings (long format) -----------------------------------------
    # Funds prefer large caps and within-industry concentration.
    # Each fund holds 30-60 stocks.  Holdings are persistent across time.
    fund_records: list[tuple[str, str, float]] = []
    for f in range(n_funds):
        fund_id = f"F{str(f).zfill(3)}"
        n_hold = rng.integers(30, 60)
        # Bias: 60% weight on top-quartile market cap
        cap_order = np.argsort(-market_cap.values)
        bias_n = int(0.6 * n_hold)
        biased_idx = rng.choice(cap_order[: max(bias_n, n_hold // 2)], size=bias_n, replace=False)
        # Remaining: industry-clustered picks
        remaining_n = n_hold - bias_n
        ind_pick = rng.integers(0, n_ind)
        mask = industry_codes == ind_pick
        if mask.sum() == 0:
            other_idx = rng.choice(n_stocks, size=remaining_n, replace=False)
        else:
            other_idx = rng.choice(np.where(mask)[0], size=min(remaining_n, mask.sum()), replace=False)
        chosen = np.unique(np.concatenate([biased_idx, other_idx]))
        weights = rng.dirichlet(np.ones(len(chosen)) * 2.0)  # moderate concentration
        for s_idx, w in zip(chosen, weights):
            fund_records.append((fund_id, stock_ids[s_idx], float(w)))

    fund_holdings = pd.DataFrame(fund_records, columns=["fund_id", "stock_id", "weight"])

    # ---- Market regime (3-state) --------------------------------------------
    # A simple hidden Markov-style sequence: bull, bear, sideways
    # Driven by rolling sign of market factor
    market_rolling = factor_df["market"].rolling(60).mean().fillna(0)
    regime = pd.Series("sideways", index=trading_days, name="regime")
    regime[market_rolling > 0.001] = "bull"
    regime[market_rolling < -0.001] = "bear"

    return SyntheticDataBundle(
        prices=prices,
        returns=returns_df,
        fundamentals=fundamentals,
        fund_holdings=fund_holdings,
        stock_industry=stock_industry,
        market_regime=regime,
        trading_days=trading_days,
    )
