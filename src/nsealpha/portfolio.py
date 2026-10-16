"""Turn forecasts into a market-neutral book and measure it after costs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def neutral_weights(score: pd.DataFrame, exposures: list[pd.DataFrame], universe: pd.DataFrame,
                    halflife: float | None = 5.0) -> pd.DataFrame:
    """Dollar-neutral weights (gross 1) proportional to the forecast, with exposures projected out.

    The forecast is first smoothed with an exponential moving average to cut turnover.
    """
    s = score.where(universe)
    if halflife:
        s = s.ewm(halflife=halflife, min_periods=1).mean().where(universe)
    S = s.to_numpy()
    W = np.zeros_like(S, dtype="float64")
    E = np.stack([e.to_numpy() for e in exposures], axis=-1)
    for t in range(len(s)):
        rows = ~np.isnan(S[t]) & ~np.isnan(E[t]).any(axis=1)
        if rows.sum() < 30:
            continue
        X = np.column_stack([np.ones(rows.sum()), E[t][rows]])
        y = S[t][rows]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ coef
        gross = np.abs(r).sum()
        if gross > 0:
            W[t, rows] = r / gross
    return pd.DataFrame(W, index=s.index, columns=s.columns)


def backtest(weights: pd.DataFrame, daily_ret: pd.DataFrame, cost_bps: float, lag: int = 1) -> pd.DataFrame:
    """Daily P&L of weights decided at close t, traded at close t+lag, held to the next close.

    Costs are charged on traded notional (sum of absolute weight changes) at cost_bps per side.
    """
    held = weights.shift(lag).fillna(0.0)
    gross = (held.shift(1) * daily_ret.fillna(0.0)).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    cost = turnover * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "turnover": turnover})


def perf_stats(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    ann_ret, ann_vol = r.mean() * 252, r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
        "max_drawdown": float(dd),
        "hit_rate": float((r > 0).mean()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurt() + 3),
        "n_days": int(len(r)),
    }


def expected_max_sharpe(sharpe_trials: np.ndarray) -> float:
    """Expected maximum Sharpe (per period) of N trials under the null of zero skill.

    Bailey and Lopez de Prado (2014), using the variance of the observed trial Sharpes.
    """
    n = len(sharpe_trials)
    if n < 2:
        return 0.0
    gamma = 0.5772156649
    sd = np.std(sharpe_trials, ddof=1)
    return float(sd * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e))))


def probabilistic_sharpe(r: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Probability that the true per-period Sharpe exceeds benchmark_sr, adjusting for
    sample length, skewness and fat tails (Bailey and Lopez de Prado, 2012)."""
    r = r.dropna()
    sr = r.mean() / r.std()
    skew, kurt = r.skew(), r.kurt() + 3
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(norm.cdf((sr - benchmark_sr) * np.sqrt(len(r) - 1) / denom))


def deflated_sharpe(r: pd.Series, sharpe_trials: np.ndarray) -> float:
    """Probabilistic Sharpe against the best Sharpe expected from luck across all trials tried."""
    return probabilistic_sharpe(r, expected_max_sharpe(np.asarray(sharpe_trials)))
