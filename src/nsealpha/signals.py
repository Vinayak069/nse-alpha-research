"""Signal library. Every signal at date t uses only data observed at or before the close of t."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Panel = dict[str, pd.DataFrame]


def market_return(panel: Panel, universe: pd.DataFrame) -> pd.Series:
    """Equal-weighted daily return of the universe."""
    return panel["ret"].where(universe).mean(axis=1)


def rolling_beta(ret: pd.DataFrame, mkt: pd.Series, window: int) -> pd.DataFrame:
    minp = window // 2
    m = pd.DataFrame(np.repeat(mkt.values[:, None], ret.shape[1], axis=1), index=ret.index, columns=ret.columns)
    m = m.where(ret.notna())
    mean_r = ret.rolling(window, min_periods=minp).mean()
    mean_m = m.rolling(window, min_periods=minp).mean()
    cov = (ret * m).rolling(window, min_periods=minp).mean() - mean_r * mean_m
    var = (m * m).rolling(window, min_periods=minp).mean() - mean_m**2
    return cov / var


def _ratio(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a / b.where(b != 0)


def compute_signals(panel: Panel, universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    o, h, lo, c = panel["Open"], panel["High"], panel["Low"], panel["Close"]
    vol, trades, dlv, r = panel["Volume"], panel["TOTAL_TRADES"], panel["DLV_QTY"], panel["ret"]
    value = panel["value"]
    mkt = market_return(panel, universe)
    lr = np.log1p(r)

    def past_ret(a: int, b: int) -> pd.DataFrame:
        """Compounded return from close t-a to close t-b (a > b)."""
        return np.expm1(lr.rolling(a - b, min_periods=(a - b) * 3 // 4).sum().shift(b))

    beta252 = rolling_beta(r, mkt, 252)
    beta60 = rolling_beta(r, mkt, 60)
    resid = r - beta252.shift(1).mul(mkt, axis=0)
    hl = np.log(_ratio(h, lo))
    dlv_pct = _ratio(dlv, vol)
    trade_size = _ratio(vol, trades)
    overnight = np.log(_ratio(o, c.shift(1)))
    intraday = np.log(_ratio(c, o))

    s: dict[str, Callable[[], pd.DataFrame]] = {
        # momentum and reversal
        "mom_12_1": lambda: past_ret(252, 21),
        "mom_6_1": lambda: past_ret(126, 21),
        "mom_3_1": lambda: past_ret(63, 21),
        "resid_mom_12_1": lambda: resid.rolling(231, min_periods=150).sum().shift(21),
        "ret_1d": lambda: r,
        "ret_5d": lambda: past_ret(5, 0),
        "ret_21d": lambda: past_ret(21, 0),
        "seasonal_12m": lambda: past_ret(252, 231),
        "high_52w": lambda: c / h.rolling(252, min_periods=126).max(),
        "ma50_gap": lambda: c / c.rolling(50, min_periods=40).mean() - 1,
        "ma200_gap": lambda: c / c.rolling(200, min_periods=150).mean() - 1,
        # risk
        "vol_20": lambda: r.rolling(20, min_periods=15).std(),
        "vol_60": lambda: r.rolling(60, min_periods=45).std(),
        "beta_252": lambda: beta252,
        "idio_vol_60": lambda: resid.rolling(60, min_periods=45).std(),
        "max_21": lambda: r.rolling(21, min_periods=15).max(),
        "skew_60": lambda: r.rolling(60, min_periods=45).skew(),
        "range_vol_20": lambda: np.sqrt((hl**2).rolling(20, min_periods=15).mean() / (4 * np.log(2))),
        "beta_change": lambda: beta60 - beta252,
        # liquidity and trading activity
        "log_value_63": lambda: np.log(value.rolling(63, min_periods=40).median()),
        "amihud_21": lambda: np.log(_ratio(r.abs(), value).rolling(21, min_periods=15).mean() + 1e-12),
        "volume_shock": lambda: np.log(_ratio(vol.rolling(5, min_periods=3).mean(), vol.rolling(63, min_periods=40).mean())),
        # intraday structure
        "overnight_21": lambda: overnight.rolling(21, min_periods=15).sum(),
        "intraday_21": lambda: intraday.rolling(21, min_periods=15).sum(),
        "clv_5": lambda: _ratio((c - lo) - (h - c), h - lo).rolling(5, min_periods=3).mean(),
        # NSE delivery data (available from Oct 2019)
        "delivery_pct_20": lambda: dlv_pct.rolling(20, min_periods=15).mean(),
        "delivery_shock": lambda: dlv_pct.rolling(5, min_periods=3).mean() - dlv_pct.rolling(63, min_periods=40).mean(),
        "trade_size_shock": lambda: np.log(_ratio(trade_size.rolling(5, min_periods=3).mean(), trade_size.rolling(63, min_periods=40).mean())),
    }
    out = {}
    for name, fn in s.items():
        x = fn().astype("float32")
        out[name] = x.replace([np.inf, -np.inf], np.nan).where(universe)
    return out


# Exposures every signal and the target are neutralised against.
RISK_FACTORS = ["beta_252", "log_value_63"]
