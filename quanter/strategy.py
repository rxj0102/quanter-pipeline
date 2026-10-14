"""Baseline strategy: trend-filtered, volatility-targeted exposure to one asset,
with a drawdown stop.

This is a placeholder example so the pipeline has something honest to run.
Replace `compute_signals` with your own research strategy. The rule that must
hold for any replacement: the exposure decided at the close of day t may only
use information available at the close of day t, and it is applied to the
return of day t+1.

The drawdown stop obeys the same rule. It looks at the strategy's own net
equity through the close of day t, and if the stop engages the position on
day t+1 is zero.
"""
import numpy as np
import pandas as pd

from .config import Config


def compute_signals(prices: pd.Series, cfg: Config) -> pd.DataFrame:
    rets = prices.pct_change()
    vol = rets.rolling(cfg.vol_window).std() * np.sqrt(cfg.trading_days)
    sma = prices.rolling(cfg.trend_window).mean()
    risk_on = prices > sma
    raw = (cfg.target_vol / vol).clip(upper=cfg.max_leverage)
    exposure = raw.where(risk_on, 0.0)
    out = pd.DataFrame({"vol": vol, "sma": sma, "risk_on": risk_on, "exposure": exposure})
    return out.dropna()


def run_strategy(prices: pd.Series, cfg: Config) -> pd.DataFrame:
    """Returns one row per day with the position held, costs and net return."""
    sig = compute_signals(prices, cfg)
    asset_ret = prices.pct_change().reindex(sig.index)
    n = len(sig)
    sig_exp = sig["exposure"].to_numpy(dtype=float)
    ret = asset_ret.to_numpy(dtype=float)

    position = np.full(n, np.nan)
    gross = np.full(n, np.nan)
    cost = np.full(n, np.nan)
    net = np.full(n, np.nan)
    strat_dd = np.full(n, np.nan)
    stop_active = np.zeros(n, dtype=bool)  # the stop forced the position to zero today
    stopped_after = np.zeros(n, dtype=bool)  # the stop is engaged after today's close
    trigger = np.zeros(n, dtype=bool)  # the stop engaged at today's close

    stopped = False
    counter = 0
    equity = 1.0
    peak = 1.0
    prev_pos = 0.0

    for i in range(1, n):
        stop_active[i] = stopped
        p = 0.0 if stopped else sig_exp[i - 1]  # decided at yesterday's close
        c = abs(p - prev_pos) * cfg.cost_bps / 1e4
        g = p * ret[i]
        r = g - c

        equity *= 1.0 + r
        peak = max(peak, equity)
        dd = equity / peak - 1.0

        if stopped:
            counter -= 1
            if counter <= 0:
                stopped = False
                peak = equity  # re-arm measuring from here
        elif cfg.dd_stop > 0 and dd <= -cfg.dd_stop:
            stopped = True
            counter = cfg.dd_cooldown
            trigger[i] = True

        position[i], gross[i], cost[i], net[i], strat_dd[i] = p, g, c, r, dd
        stopped_after[i] = stopped
        prev_pos = p

    out = pd.DataFrame(
        {
            "asset_ret": ret,
            "position": position,
            "gross": gross,
            "cost": cost,
            "net": net,
            "regime": np.where(sig["risk_on"], "risk-on", "risk-off"),
            "signal_exposure": sig_exp,
            "stop_active": stop_active,
            "stopped_after_close": stopped_after,
            "stop_trigger": trigger,
            "strategy_dd": strat_dd,
        },
        index=sig.index,
    )
    return out.dropna(subset=["net"])
