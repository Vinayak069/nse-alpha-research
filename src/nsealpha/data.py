"""Load NSE end-of-day data into wide (date x symbol) panels and build a liquid universe."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["Open", "High", "Low", "Close", "Volume", "TOTAL_TRADES", "DLV_QTY"]

# NSE price bands cap most stocks at 20% a day; F&O names have no band but a
# close-to-close move beyond 40% in a liquid stock is almost always a bad print
# or an unadjusted corporate action, so it is dropped rather than traded on.
MAX_ABS_DAILY_RETURN = 0.40


def load_panel(data_dir: str | Path, start: str = "2010-01-01", cache: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Read every <symbol>.csv in data_dir and pivot each field to a date x symbol frame."""
    if cache is not None and Path(cache).exists():
        long = pd.read_parquet(cache)
    else:
        frames = []
        for path in sorted(Path(data_dir).glob("*.csv")):
            df = pd.read_csv(path, usecols=["Date", *FIELDS], parse_dates=["Date"])
            df = df[df["Date"] >= start]
            if df.empty:
                continue
            df["symbol"] = path.stem.upper()
            frames.append(df)
        long = pd.concat(frames, ignore_index=True)
        long[FIELDS] = long[FIELDS].astype("float32")
        long = long.drop_duplicates(["Date", "symbol"], keep="last")
        if cache is not None:
            long.to_parquet(cache)
    panel = {f: long.pivot(index="Date", columns="symbol", values=f).sort_index() for f in FIELDS}
    return clean_panel(panel)


def clean_panel(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Mask non-positive prices and implausible jumps; add returns and traded value."""
    p = {k: v.copy() for k, v in panel.items()}
    for f in ["Open", "High", "Low", "Close"]:
        p[f] = p[f].where(p[f] > 0)
    ret = p["Close"].pct_change(fill_method=None)
    bad = ret.abs() > MAX_ABS_DAILY_RETURN
    p["ret"] = ret.mask(bad)
    p["value"] = p["Close"] * p["Volume"]
    return p


def build_universe(panel: dict[str, pd.DataFrame], top_n: int = 500, min_price: float = 20.0,
                   min_history: int = 252, liq_window: int = 63) -> pd.DataFrame:
    """Boolean date x symbol mask of the top_n most liquid stocks.

    Membership is decided on the first trading day of each month from trailing data only
    (median traded value over liq_window days, price floor, listing age) and held for the
    month. A member must also have a valid close on the day itself.
    """
    close = panel["Close"]
    liq = panel["value"].rolling(liq_window, min_periods=liq_window // 2).median()
    age = close.notna().cumsum()
    eligible = (close >= min_price) & (age >= min_history)
    score = liq.where(eligible)
    month_start = ~close.index.to_period("M").duplicated()
    ranks = score[month_start].rank(axis=1, ascending=False)
    members = (ranks <= top_n).reindex(close.index).ffill().fillna(False).astype(bool)
    return members & close.notna()


def forward_returns(panel: dict[str, pd.DataFrame], horizon: int, lag: int = 1) -> pd.DataFrame:
    """Return from close t+lag to close t+lag+horizon, aligned to date t.

    Built by compounding the cleaned daily returns, so masked bad prints never leak
    into a target. lag=1 means a signal computed at the close of t is traded at the close of t+1,
    so the same-day close is never assumed to be tradable.
    """
    log_ret = np.log1p(panel["ret"])
    return np.expm1(log_ret.rolling(horizon, min_periods=horizon).sum().shift(-(lag + horizon)))
