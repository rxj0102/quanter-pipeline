import json
import os

import numpy as np
import pandas as pd

from quanter import metrics as m
from quanter.config import Config
from quanter.data import synthetic_prices
from quanter.decay import decay_check
from quanter.pipeline import build_results, label_segments, update_paper_log
from quanter.strategy import run_strategy

CFG = Config()


def test_drawdown_known_series():
    r = pd.Series([0.10, -0.50, 0.20])
    # equity: 1.1, 0.55, 0.66 ; peak 1.1 ; worst drawdown = 0.55/1.1 - 1 = -0.5
    assert abs(m.max_drawdown(r) - (-0.5)) < 1e-12


def test_var_cvar_known_returns():
    r = pd.Series(np.linspace(-0.10, 0.10, 201))
    var, cvar = m.hist_var_cvar(r, window=201, level=0.95)
    assert 0.08 < var < 0.095
    assert cvar > var


def test_no_lookahead_in_positions():
    prices = synthetic_prices()
    base = run_strategy(prices, CFG)
    cut = prices.index[2500]
    altered = prices.copy()
    altered.loc[altered.index > cut] *= 1.5  # change the future only
    new = run_strategy(altered, CFG)
    common = base.index[base.index <= cut]
    pd.testing.assert_series_equal(base.loc[common, "position"], new.loc[common, "position"])


def test_costs_reduce_returns():
    prices = synthetic_prices()
    # The stop is path dependent, so switch it off to compare costs cleanly.
    free = run_strategy(prices, Config(cost_bps=0.0, dd_stop=0.0))["net"].sum()
    costly = run_strategy(prices, Config(cost_bps=10.0, dd_stop=0.0))["net"].sum()
    assert costly < free


def test_exposure_respects_leverage_cap():
    df = run_strategy(synthetic_prices(), CFG)
    assert df["position"].max() <= CFG.max_leverage + 1e-12
    assert df["position"].min() >= 0.0


def test_segments_are_ordered_and_disjoint():
    idx = pd.bdate_range("2018-01-01", "2026-12-31")
    seg = label_segments(idx, CFG)
    assert seg[idx <= pd.Timestamp(CFG.in_sample_end)].eq("in_sample").all()
    assert seg[idx >= pd.Timestamp(CFG.paper_start)].eq("paper").all()
    mid = seg[(idx > pd.Timestamp(CFG.in_sample_end)) & (idx < pd.Timestamp(CFG.paper_start))]
    assert mid.eq("out_of_sample").all()


def test_results_schema_and_synthetic_flag():
    res = build_results(synthetic_prices(), CFG, synthetic=True)
    res.pop("_frame")
    res.pop("_segments_series")
    assert res["meta"]["synthetic"] is True
    for key in (
        "rolling_sharpe",
        "max_drawdown_all",
        "var_95_1d",
        "cvar_95_1d",
        "regime",
        "paper_days",
        "drawdown_stop",
        "decay",
    ):
        assert key in res["summary"]
    n = len(res["series"]["dates"])
    for key in ("segment", "cum_return", "drawdown", "rolling_sharpe"):
        assert len(res["series"][key]) == n
    json.dumps(res)  # must be serializable, no NaN objects


def _crash_prices():
    n = 420
    idx = pd.bdate_range("2020-01-01", periods=n)
    rets = 0.001 + 0.001 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    rets[300:305] = -0.04  # five straight 4 percent drops
    return pd.Series(100 * np.cumprod(1 + rets), index=idx, name="CRASH")


def _always_long(prices, cfg):
    """Stand-in signal that ignores the trend filter so the stop is what we test."""
    idx = prices.index[1:]
    return pd.DataFrame({"vol": 0.1, "sma": 1.0, "risk_on": True, "exposure": 1.0}, index=idx)


def test_drawdown_stop_engages_and_holds_for_cooldown():
    import quanter.strategy as strat

    original = strat.compute_signals
    strat.compute_signals = _always_long
    try:
        cfg = Config(dd_stop=0.15, dd_cooldown=21, cost_bps=0.0)
        df = run_strategy(_crash_prices(), cfg)
    finally:
        strat.compute_signals = original

    assert int(df["stop_trigger"].sum()) == 1
    trig = df.index[df["stop_trigger"]][0]
    # The day the stop engages is still held at the old position: it acts on the next day.
    assert df.loc[trig, "position"] > 0
    after = df.loc[df.index > trig]
    held = after.iloc[:21]
    assert held["stop_active"].all()
    assert (held["position"] == 0.0).all()
    # After the cooldown the strategy is back in the market.
    resumed = after.iloc[21]
    assert not resumed["stop_active"]
    assert resumed["position"] > 0


def test_drawdown_stop_disabled():
    df = run_strategy(synthetic_prices(), Config(dd_stop=0.0))
    assert not df["stop_active"].any()
    assert not df["stop_trigger"].any()


def _decay_inputs(seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=800)
    net = pd.Series(rng.normal(0.002, 0.01, len(idx)), index=idx)
    seg = pd.Series("out_of_sample", index=idx)
    seg.iloc[:100] = "in_sample"
    seg.iloc[-100:] = "paper"
    return net, seg


def test_decay_flags_a_collapse():
    net, seg = _decay_inputs()
    net.iloc[-126:] = np.random.default_rng(9).normal(-0.004, 0.01, 126)
    res = decay_check(net, seg, window=126)
    assert res["status"] == "review"
    assert "sharpe_below_validated_range" in res["flags"]


def test_decay_within_range_when_behaviour_is_typical():
    net, seg = _decay_inputs()
    rs = m.rolling_sharpe(net, 126)
    ref = rs[seg == "out_of_sample"].dropna()
    j = net.index.get_loc((ref - ref.median()).abs().idxmin())
    net.iloc[-126:] = net.iloc[j - 125 : j + 1].values
    res = decay_check(net, seg, window=126)
    assert res["status"] == "within_range", res


def test_decay_needs_enough_reference_data():
    net, seg = _decay_inputs()
    seg[:] = "in_sample"
    assert decay_check(net, seg, window=126)["status"] == "insufficient_data"


def test_decay_flat_strategy():
    net, seg = _decay_inputs()
    net.iloc[-126:] = 0.0
    assert decay_check(net, seg, window=126)["status"] == "flat"


def test_paper_log_is_append_only(tmp_path):
    prices = synthetic_prices(end="2026-10-30")
    cfg = Config(paper_start="2026-10-12")
    res = build_results(prices, cfg, synthetic=True)
    df, seg = res["_frame"], res["_segments_series"]
    log = str(tmp_path / "paper_log.csv")

    assert update_paper_log(df, seg, log) == []
    first = pd.read_csv(log)
    assert len(first) > 0

    # Re-running with identical data changes nothing.
    assert update_paper_log(df, seg, log) == []
    assert pd.read_csv(log).equals(first)

    # A tampered recomputation is reported and the logged rows are not overwritten.
    tampered = df.copy()
    tampered.loc[tampered.index[-1], "net"] += 0.01
    mismatched = update_paper_log(tampered, seg, log)
    assert len(mismatched) == 1
    assert pd.read_csv(log).equals(first)
