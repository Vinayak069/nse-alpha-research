"""End-to-end study: signals -> IC analysis -> walk-forward models -> neutral portfolio -> holdout.

Usage: python scripts/run_research.py --data ../eod2_data/daily
Results go to docs/results.json and docs/figures/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nsealpha.data import build_universe, forward_returns, load_panel
from nsealpha.evaluation import ic_decay, ic_summary
from nsealpha.model import WalkForwardConfig, stack, walk_forward
from nsealpha.portfolio import backtest, deflated_sharpe, neutral_weights, perf_stats, probabilistic_sharpe
from nsealpha.preprocess import neutralize
from nsealpha.signals import RISK_FACTORS, compute_signals

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 5
RESEARCH_END = "2022-12-31"   # everything after this is the untouched holdout
HOLDOUT_START = "2023-01-01"
BASE_COST_BPS = 10.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def period(x, start=None, end=None):
    return x.loc[start:end]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--cache", default=str(ROOT / "data" / "panel.parquet"))
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. top200")
    args = ap.parse_args()
    (ROOT / "data").mkdir(exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    figs = ROOT / "docs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    results: dict = {"config": {"horizon_days": HORIZON, "top_n": args.top_n, "research_end": RESEARCH_END,
                                "holdout_start": HOLDOUT_START, "base_cost_bps_per_side": BASE_COST_BPS}}

    log("loading panel")
    panel = load_panel(args.data, start=args.start, cache=args.cache)
    universe = build_universe(panel, top_n=args.top_n)
    keep = universe.any(axis=0)
    panel = {k: v.loc[:, keep] for k, v in panel.items()}
    universe = universe.loc[:, keep]
    results["data"] = {"first_date": str(universe.index[0].date()), "last_date": str(universe.index[-1].date()),
                       "symbols_ever_in_universe": int(keep.sum()), "avg_names_per_day": float(universe.sum(axis=1)[universe.sum(axis=1) > 0].mean())}
    log(f"universe: {keep.sum()} symbols")

    log("computing signals")
    sig = compute_signals(panel, universe)
    ret = panel["ret"].where(universe)
    fwd = forward_returns(panel, HORIZON).where(universe)

    log("raw IC table (research period)")
    raw_ic = {k: ic_summary(period(v, end=RESEARCH_END), period(fwd, end=RESEARCH_END), HORIZON) for k, v in sig.items()}

    log("neutralising")
    exposures = {k: sig[k] for k in RISK_FACTORS}
    feats = {k: v for k, v in sig.items() if k not in RISK_FACTORS}
    neut = neutralize(feats, exposures, universe)
    neut_ic = {k: ic_summary(period(v, end=RESEARCH_END), period(fwd, end=RESEARCH_END), HORIZON) for k, v in neut.items()}
    results["ic_raw"] = raw_ic
    results["ic_neutral"] = neut_ic

    ranked = sorted(neut_ic, key=lambda k: -abs(neut_ic[k]["t_stat_nw"]))
    log("IC decay for top signals")
    decay = {k: ic_decay(period(neut[k], end=RESEARCH_END), period(ret, end=RESEARCH_END), lags=[0, 1, 2, 4, 9, 19, 39]).to_dict()
             for k in ranked[:6]}
    results["ic_decay"] = decay

    log("building target and training table")
    target = neutralize({"target": fwd}, exposures, universe)["target"]
    table = stack(neut, target)
    names = list(neut)

    trials = {
        "ic_weighted": WalkForwardConfig(model="ic_weighted"),
        "ridge_a10": WalkForwardConfig(model="ridge", ridge_alpha=10),
        "ridge_a100": WalkForwardConfig(model="ridge", ridge_alpha=100),
        "ridge_a1000": WalkForwardConfig(model="ridge", ridge_alpha=1000),
        "gbm": WalkForwardConfig(model="gbm"),
    }
    halflives = [2.0, 5.0, 10.0]
    expo_list = [exposures[k].where(universe) for k in RISK_FACTORS]
    trial_rows, books, weights_by_model = [], {}, {}
    for name, cfg in trials.items():
        cfg.first_test_year = 2015
        log(f"walk-forward {name}")
        fc, w = walk_forward(table, names, cfg)
        weights_by_model[name] = w
        score = fc.unstack().reindex(index=universe.index, columns=universe.columns)
        for hl in halflives:
            wts = neutral_weights(score, expo_list, universe, halflife=hl)
            bt = backtest(wts, panel["ret"], BASE_COST_BPS)
            val = period(bt, "2015-01-01", RESEARCH_END)
            st = perf_stats(val["net"])
            trial_rows.append({"model": name, "halflife": hl, **{f"val_{k}": v for k, v in st.items()},
                               "val_turnover": float(val["turnover"].mean()),
                               "val_daily_sharpe": float(val["net"].mean() / val["net"].std())})
            books[(name, hl)] = (wts, bt)
    trials_df = pd.DataFrame(trial_rows).sort_values("val_sharpe", ascending=False)
    results["trials"] = trials_df.to_dict(orient="records")
    best = trials_df.iloc[0]
    best_key = (best["model"], best["halflife"])
    log(f"selected {best_key} on validation")

    wts, bt = books[best_key]
    val = period(bt, "2015-01-01", RESEARCH_END)
    hold = period(bt, HOLDOUT_START)
    trial_sr = trials_df["val_daily_sharpe"].to_numpy()
    results["selected"] = {
        "model": best["model"], "halflife": float(best["halflife"]),
        "validation": {**perf_stats(val["net"]), "gross_sharpe": perf_stats(val["gross"])["sharpe"],
                       "turnover": float(val["turnover"].mean()),
                       "deflated_sharpe_prob": deflated_sharpe(val["net"], trial_sr), "n_trials": len(trial_sr)},
        "holdout": {**perf_stats(hold["net"]), "gross_sharpe": perf_stats(hold["gross"])["sharpe"],
                    "turnover": float(hold["turnover"].mean()),
                    "prob_sharpe_gt_0": probabilistic_sharpe(hold["net"])},
        "cost_sensitivity_holdout": {str(c): perf_stats(period(backtest(wts, panel["ret"], c), HOLDOUT_START)["net"])["sharpe"]
                                     for c in [0, 5, 10, 20, 30]},
        "model_weights_last_fit": {k: float(v) for k, v in weights_by_model[best["model"]].iloc[-1].dropna().items()}
        if best["model"] != "gbm" else {},
    }
    yearly = bt["net"].groupby(bt.index.year).apply(lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan)
    results["selected"]["sharpe_by_year"] = {int(y): float(v) for y, v in yearly.loc[2015:].items()}

    hold_ic = {k: ic_summary(period(neut[k], HOLDOUT_START), period(fwd, HOLDOUT_START), HORIZON)["ic_mean"] for k in ranked[:8]}
    results["holdout_ic_top_signals"] = hold_ic

    log("figures")
    ic_df = pd.DataFrame(neut_ic).T.sort_values("ic_mean")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(ic_df.index, ic_df["ic_mean"], color=["#c0504d" if v < 0 else "#4f81bd" for v in ic_df["ic_mean"]])
    ax.set_xlabel(f"mean rank IC vs {HORIZON}-day forward return (beta/size neutral, 2011-2022)")
    ax.axvline(0, color="k", lw=0.6)
    fig.tight_layout(); fig.savefig(figs / f"ic_by_signal{sfx}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, d in decay.items():
        ax.plot(list(d.keys()), list(d.values()), marker="o", label=k)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("days after trade date"); ax.set_ylabel("mean rank IC (1-day return)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figs / f"ic_decay{sfx}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    curve = period(bt, "2015-01-01")
    ax.plot(curve.index, curve["gross"].cumsum(), label="gross", lw=1)
    ax.plot(curve.index, curve["net"].cumsum(), label=f"net of {BASE_COST_BPS:.0f} bps/side", lw=1)
    ax.axvline(pd.Timestamp(HOLDOUT_START), color="grey", ls="--", lw=0.8)
    ax.text(pd.Timestamp(HOLDOUT_START), ax.get_ylim()[1] * 0.9, " holdout", color="grey")
    ax.set_ylabel("cumulative return (sum, gross exposure 1)")
    ax.set_title(f"{best['model']} forecast, EMA half-life {best['halflife']:.0f}d")
    ax.legend(); fig.tight_layout(); fig.savefig(figs / f"equity_curve{sfx}.png", dpi=130); plt.close(fig)

    (ROOT / "docs" / f"results{sfx}.json").write_text(json.dumps(results, indent=2, default=float))
    log("done")


if __name__ == "__main__":
    main()
