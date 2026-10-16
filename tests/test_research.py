import numpy as np
import pandas as pd
import pytest

from conftest import make_panel
from nsealpha.data import build_universe, clean_panel, forward_returns
from nsealpha.evaluation import newey_west_tstat, rank_ic
from nsealpha.model import WalkForwardConfig, stack, walk_forward
from nsealpha.portfolio import (backtest, deflated_sharpe, expected_max_sharpe, neutral_weights,
                                probabilistic_sharpe)
from nsealpha.preprocess import neutralize, rank_gauss
from nsealpha.signals import compute_signals


def test_signals_never_use_future_data():
    base = make_panel(seed=1)
    cut = 300
    universe = build_universe(base, top_n=40, min_history=50)
    before = compute_signals(base, universe)

    raw = {k: base[k].copy() for k in ["Open", "High", "Low", "Close", "Volume", "TOTAL_TRADES", "DLV_QTY"]}
    noise = np.random.default_rng(9).uniform(0.5, 2.0, size=raw["Close"].iloc[cut + 1:].shape)
    for k in raw:
        raw[k].iloc[cut + 1:] *= noise  # scramble every price and volume after the cut
    shocked = clean_panel(raw)
    after = compute_signals(shocked, build_universe(shocked, top_n=40, min_history=50))

    for name in before:
        a, b = before[name].iloc[: cut + 1], after[name].iloc[: cut + 1]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-5, obj=name)


def test_universe_membership_is_monthly_and_limited(panel):
    u = build_universe(panel, top_n=20, min_history=50)
    assert u.sum(axis=1).max() <= 20
    month = u.index.to_period("M")
    for m in month.unique()[3:6]:
        block = u[month == m]
        assert (block.eq(block.iloc[0])).all().all()


def test_forward_return_alignment(panel):
    fwd = forward_returns(panel, horizon=3, lag=1)
    c = panel["Close"]
    t = 100
    expected = c.iloc[t + 4] / c.iloc[t + 1] - 1
    np.testing.assert_allclose(fwd.iloc[t], expected, rtol=1e-5)


def test_rank_gauss_is_standardised():
    x = pd.DataFrame(np.random.default_rng(0).exponential(size=(5, 400)))
    z = rank_gauss(x)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1, atol=0.05)


def test_neutralize_removes_exposure():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = range(300)
    expo = pd.DataFrame(rng.normal(size=(5, 300)), idx, cols)
    feat = expo * 2 + pd.DataFrame(rng.normal(size=(5, 300)), idx, cols)
    uni = pd.DataFrame(True, idx, cols)
    out = neutralize({"f": feat}, {"e": expo}, uni)["f"]
    ez = rank_gauss(expo)
    for t in range(5):
        assert abs(np.corrcoef(out.iloc[t], ez.iloc[t])[0, 1]) < 1e-8


def test_rank_ic_perfect_and_inverse():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(10, 50)))
    assert np.allclose(rank_ic(x, x * 2 + 1), 1.0)
    assert np.allclose(rank_ic(x, -x), -1.0)


def test_newey_west_tstat_shrinks_with_autocorrelation():
    rng = np.random.default_rng(0)
    e = rng.normal(size=2000)
    ar = np.convolve(e, np.ones(10), mode="valid") / 10 + 0.05
    s = pd.Series(ar)
    assert newey_west_tstat(s, lags=10) < newey_west_tstat(s, lags=0)


def test_backtest_cannot_profit_from_same_day_information():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    ret = pd.DataFrame(rng.normal(0, 0.01, size=(300, 40)), idx)
    # a "signal" that knows the next day's return would be look-ahead; the lag must neutralise it
    cheat = ret.shift(-1)
    w = cheat.sub(cheat.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1), axis=0).fillna(0)
    assert abs(backtest(w, ret, cost_bps=0)["gross"].mean()) < 5e-4
    # a signal that genuinely forecasts the return two days after decision should profit
    honest = ret.shift(-2)
    w2 = honest.sub(honest.mean(axis=1), axis=0)
    w2 = w2.div(w2.abs().sum(axis=1), axis=0).fillna(0)
    assert backtest(w2, ret, cost_bps=0)["gross"].mean() > 1e-3


def test_costs_scale_with_turnover():
    idx = pd.bdate_range("2020-01-01", periods=4)
    w = pd.DataFrame([[0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, 0.5]], idx)
    ret = pd.DataFrame(0.0, idx, [0, 1])
    bt = backtest(w, ret, cost_bps=10)
    assert bt["turnover"].tolist() == [0.0, 1.0, 0.0, 2.0]
    assert np.isclose(bt["cost"].sum(), 3.0 * 10 / 1e4)


def test_neutral_weights_are_dollar_and_exposure_neutral(panel):
    u = build_universe(panel, top_n=50, min_history=30)
    sig = compute_signals(panel, u)
    w = neutral_weights(sig["mom_3_1"], [sig["beta_252"].where(u)], u, halflife=None)
    live = w.abs().sum(axis=1) > 0
    assert live.any()
    assert np.allclose(w[live].sum(axis=1), 0, atol=1e-9)
    assert np.allclose(w[live].abs().sum(axis=1), 1)
    beta = sig["beta_252"].fillna(0)
    assert np.allclose((w[live] * beta[live]).sum(axis=1), 0, atol=1e-8)


def test_walk_forward_trains_only_on_the_past():
    idx = pd.bdate_range("2010-01-01", "2016-12-31")
    cols = range(40)
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(size=(len(idx), 40)), idx, cols)
    target = f * 0.1 + pd.DataFrame(rng.normal(size=(len(idx), 40)), idx, cols)
    table = stack({"f": f}, target)
    cfg = WalkForwardConfig(model="ridge", first_test_year=2013, min_train_years=3, embargo_days=10)
    fc, weights = walk_forward(table, ["f"], cfg)
    assert fc.index.get_level_values(0).min() >= pd.Timestamp("2013-01-01")
    assert list(weights.index) == [2013, 2014, 2015, 2016]
    assert (weights["f"] > 0).all()


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, size=1500))
    few = rng.normal(0, 0.01, size=3)
    many = rng.normal(0, 0.03, size=200)
    assert expected_max_sharpe(many) > expected_max_sharpe(few)
    assert deflated_sharpe(r, many) < deflated_sharpe(r, few) <= probabilistic_sharpe(r)
