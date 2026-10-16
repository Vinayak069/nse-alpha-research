"""Walk-forward combination of signals into one forecast."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def stack(features: dict[str, pd.DataFrame], target: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long (date, symbol) table of features, plus the target when given."""
    cols = {k: v.stack(future_stack=True) for k, v in features.items()}
    if target is not None:
        cols["target"] = target.stack(future_stack=True)
    df = pd.DataFrame(cols)
    feat_names = list(features)
    return df[df[feat_names].notna().any(axis=1)]


@dataclass
class WalkForwardConfig:
    model: str = "ridge"           # "ic_weighted", "ridge" or "gbm"
    first_test_year: int = 2016
    last_test_year: int = 2026
    min_train_years: int = 3
    embargo_days: int = 10         # gap between the last training target and the first test date
    sample_every: int = 5          # thin training dates to reduce overlap between targets
    ridge_alpha: float = 100.0
    gbm_params: dict | None = None


def _fit_predict(cfg: WalkForwardConfig, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                 names: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    if cfg.model == "gbm":
        params = dict(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                      min_samples_leaf=500, random_state=0)
        params.update(cfg.gbm_params or {})
        usable = ~np.isnan(Xtr).all(axis=0)  # e.g. delivery data before it was published
        m = HistGradientBoostingRegressor(**params).fit(Xtr[:, usable], ytr)
        return m.predict(Xte[:, usable]), {}
    Xtr0, Xte0 = np.nan_to_num(Xtr), np.nan_to_num(Xte)
    if cfg.model == "ridge":
        m = Ridge(alpha=cfg.ridge_alpha * len(ytr) / 1e4).fit(Xtr0, ytr)
        return m.predict(Xte0), dict(zip(names, m.coef_))
    if cfg.model == "ic_weighted":
        # weight each signal by its training-period correlation with the target
        w = np.array([np.corrcoef(Xtr0[:, j], ytr)[0, 1] for j in range(Xtr0.shape[1])])
        w = np.nan_to_num(w)
        return Xte0 @ w, dict(zip(names, w))
    raise ValueError(cfg.model)


def walk_forward(table: pd.DataFrame, feature_names: list[str], cfg: WalkForwardConfig
                 ) -> tuple[pd.Series, pd.DataFrame]:
    """Refit once a year on all earlier data (expanding window) and predict the next year.

    Training rows end embargo_days trading days before the test year starts, so no
    training target overlaps the test period.
    """
    dates = table.index.get_level_values(0)
    all_dates = pd.DatetimeIndex(sorted(dates.unique()))
    preds, weights = [], {}
    start_year = all_dates[0].year
    for year in range(cfg.first_test_year, cfg.last_test_year + 1):
        if year - start_year < cfg.min_train_years:
            continue
        test_start = pd.Timestamp(year=year, month=1, day=1)
        pos = all_dates.searchsorted(test_start)
        if pos >= len(all_dates):
            break
        train_end = all_dates[max(pos - cfg.embargo_days, 0)]
        train_dates = all_dates[all_dates < train_end][:: cfg.sample_every]
        tr = table[dates.isin(train_dates)].dropna(subset=["target"])
        te = table[(dates >= test_start) & (dates < pd.Timestamp(year=year + 1, month=1, day=1))]
        if tr.empty or te.empty:
            continue
        p, w = _fit_predict(cfg, tr[feature_names].to_numpy(), tr["target"].to_numpy(),
                            te[feature_names].to_numpy(), feature_names)
        preds.append(pd.Series(p, index=te.index))
        weights[year] = w
    return pd.concat(preds).rename("forecast"), pd.DataFrame(weights).T
