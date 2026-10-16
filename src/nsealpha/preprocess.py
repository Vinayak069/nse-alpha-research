"""Cross-sectional normalisation and risk neutralisation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def rank_gauss(x: pd.DataFrame) -> pd.DataFrame:
    """Map each row to standard-normal scores by rank. Robust to outliers by construction."""
    n = x.notna().sum(axis=1)
    u = (x.rank(axis=1) - 0.5).div(n, axis=0)
    return pd.DataFrame(norm.ppf(u.to_numpy(dtype="float64")), index=x.index, columns=x.columns)


def neutralize(features: dict[str, pd.DataFrame], exposures: dict[str, pd.DataFrame],
               universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rank-gauss every feature, then remove its cross-sectional projection on the exposures.

    Each day solves one least-squares problem with all features as right-hand sides.
    Missing feature values are treated as 0 (the cross-sectional mean) for the fit and
    restored as missing afterwards. Rows missing an exposure are dropped for that day.
    """
    names = list(features)
    z = {k: rank_gauss(v.where(universe)) for k, v in features.items()}
    e = {k: rank_gauss(v.where(universe)) for k, v in exposures.items()}
    dates, symbols = universe.index, universe.columns
    out = {k: np.full((len(dates), len(symbols)), np.nan, dtype="float32") for k in names}
    zs = np.stack([z[k].to_numpy() for k in names], axis=-1)       # T x N x F
    es = np.stack([e[k].to_numpy() for k in exposures], axis=-1)   # T x N x E
    uni = universe.to_numpy()
    for t in range(len(dates)):
        rows = uni[t] & ~np.isnan(es[t]).any(axis=1)
        if rows.sum() < 30:
            continue
        X = np.column_stack([np.ones(rows.sum()), es[t][rows]])
        Y = zs[t][rows]
        miss = np.isnan(Y)
        Yf = np.where(miss, 0.0, Y)
        coef, *_ = np.linalg.lstsq(X, Yf, rcond=None)
        res = Yf - X @ coef
        sd = res.std(axis=0)
        res = res / np.where(sd > 0, sd, 1.0)
        res[miss] = np.nan
        for j, k in enumerate(names):
            out[k][t, rows] = res[:, j]
    return {k: pd.DataFrame(v, index=dates, columns=symbols) for k, v in out.items()}
