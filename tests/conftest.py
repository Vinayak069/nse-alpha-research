import numpy as np
import pandas as pd
import pytest

from nsealpha.data import clean_panel


def make_panel(n_days: int = 400, n_stocks: int = 60, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    cols = [f"S{i:03d}" for i in range(n_stocks)]
    r = rng.normal(0.0003, 0.02, size=(n_days, n_stocks))
    close = 100 * np.exp(np.cumsum(r, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, size=close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, size=close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, size=close.shape)))
    volume = rng.lognormal(12, 0.5, size=close.shape)
    trades = volume / rng.uniform(20, 60, size=close.shape)
    dlv = volume * rng.uniform(0.2, 0.7, size=close.shape)
    f = lambda a: pd.DataFrame(a, index=dates, columns=cols)
    raw = {"Open": f(open_), "High": f(high), "Low": f(low), "Close": f(close), "Volume": f(volume),
           "TOTAL_TRADES": f(trades), "DLV_QTY": f(dlv)}
    return clean_panel(raw)


@pytest.fixture
def panel():
    return make_panel()
