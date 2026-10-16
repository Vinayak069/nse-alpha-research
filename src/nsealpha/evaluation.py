"""Signal evaluation: rank information coefficient, its significance and its decay."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 30) -> pd.Series:
    """Daily Spearman correlation between a signal and forward returns across stocks."""
    both = signal.notna() & fwd.notna()
    a = signal.where(both).rank(axis=1)
    b = fwd.where(both).rank(axis=1)
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    ic = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return ic.where(both.sum(axis=1) >= min_names).dropna()


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    """t-statistic of the mean of x with a Bartlett-kernel HAC variance (for overlapping returns)."""
    x = np.asarray(x.dropna(), dtype=float)
    n = len(x)
    if n < 10:
        return float("nan")
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (d[k:] @ d[:-k]) / n
    return float(x.mean() / np.sqrt(var / n))


def ic_summary(signal: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> dict[str, float]:
    ic = rank_ic(signal, fwd)
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_ir_ann": float(ic.mean() / ic.std() * np.sqrt(252 / horizon)) if ic.std() > 0 else float("nan"),
        "t_stat_nw": newey_west_tstat(ic, lags=horizon),
        "hit_rate": float((ic > 0).mean()),
        "n_days": int(len(ic)),
    }


def ic_decay(signal: pd.DataFrame, daily_ret: pd.DataFrame, lags: list[int], lag0: int = 1) -> pd.Series:
    """Mean IC of the signal at t against the single-day return k days after the trade date."""
    out = {}
    for k in lags:
        fwd = daily_ret.shift(-(lag0 + 1 + k))
        out[k] = rank_ic(signal, fwd).mean()
    return pd.Series(out, name="mean_ic")
