import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from . import metrics as m
from .config import Config
from .data import fetch_prices, synthetic_prices
from .decay import decay_check
from .strategy import run_strategy

SEGMENTS = ("in_sample", "out_of_sample", "paper")


def label_segments(index: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    in_end = pd.Timestamp(cfg.in_sample_end)
    paper = pd.Timestamp(cfg.paper_start)
    labels = pd.Series("out_of_sample", index=index)
    labels[index <= in_end] = "in_sample"
    labels[index >= paper] = "paper"
    return labels


def _clean(x):
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _r(x, nd=6):
    x = _clean(None if x is None else float(x))
    return None if x is None else round(x, nd)


def segment_stats(returns: pd.Series, cfg: Config) -> dict:
    n = int(len(returns))
    if n < 2:
        return {"days": n, "ann_return": None, "ann_vol": None, "sharpe": None, "max_drawdown": None}
    return {
        "days": n,
        "ann_return": _r(m.ann_return(returns, cfg.trading_days)),
        "ann_vol": _r(m.ann_vol(returns, cfg.trading_days)),
        "sharpe": _r(m.sharpe(returns, cfg.trading_days, cfg.risk_free_rate)),
        "max_drawdown": _r(m.max_drawdown(returns)),
    }


def build_results(prices: pd.Series, cfg: Config, synthetic: bool = False) -> dict:
    df = run_strategy(prices, cfg)
    net = df["net"]
    seg = label_segments(df.index, cfg)

    eq = m.equity_curve(net)
    dd = m.drawdown(net)
    rs = m.rolling_sharpe(net, cfg.sharpe_window, cfg.trading_days, cfg.risk_free_rate)
    var, cvar = m.hist_var_cvar(net, cfg.var_window, cfg.var_level)

    paper_net = net[seg == "paper"]
    segments = {}
    for name in SEGMENTS:
        part = net[seg == name]
        stats = segment_stats(part, cfg)
        stats["start"] = part.index[0].strftime("%Y-%m-%d") if len(part) else None
        stats["end"] = part.index[-1].strftime("%Y-%m-%d") if len(part) else None
        segments[name] = stats

    last = df.index[-1]
    stop = {
        "threshold": cfg.dd_stop,
        "cooldown_days": cfg.dd_cooldown,
        "active": bool(df["stopped_after_close"].iloc[-1]),
        "current_drawdown": _r(df["strategy_dd"].iloc[-1]),
        "trigger_count": int(df["stop_trigger"].sum()),
        "days_stopped": int(df["stop_active"].sum()),
    }
    decay = decay_check(net, seg, cfg.decay_window, cfg.decay_quantile)
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_data_date": last.strftime("%Y-%m-%d"),
            "synthetic": synthetic,
            "strategy": "baseline_trend_voltarget",
            "ticker": cfg.ticker,
            "notes": "Net of costs. Sharpe on raw returns, risk-free rate set in assumptions.",
            "assumptions": asdict(cfg),
        },
        "summary": {
            "rolling_sharpe": _r(rs.iloc[-1]),
            "max_drawdown_all": _r(m.max_drawdown(net)),
            "max_drawdown_paper": _r(m.max_drawdown(paper_net)) if len(paper_net) else None,
            "var_95_1d": _r(var),
            "cvar_95_1d": _r(cvar),
            "regime": str(df["regime"].iloc[-1]),
            "paper_days": int(len(paper_net)),
            "drawdown_stop": stop,
            "decay": decay,
        },
        "segments": segments,
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "segment": seg.tolist(),
            "cum_return": [_r(v - 1.0) for v in eq],
            "drawdown": [_r(v) for v in dd],
            "rolling_sharpe": [_r(v) for v in rs],
            "stop_active": [int(v) for v in df["stop_active"]],
        },
        "_frame": df,
        "_segments_series": seg,
    }


def update_paper_log(df: pd.DataFrame, seg: pd.Series, path: str) -> list:
    """Append-only log of paper-trading days. Existing rows are never modified.

    Returns the dates whose recomputed values differ from what was logged.
    Committing this file daily gives the git history as proof that paper
    results were not edited after the fact.
    """
    paper = df[seg == "paper"]
    cols = ["date", "position", "net_return", "regime", "stop_active"]
    new = pd.DataFrame(
        {
            "date": paper.index.strftime("%Y-%m-%d"),
            "position": paper["position"].round(6).values,
            "net_return": paper["net"].round(8).values,
            "regime": paper["regime"].values,
            "stop_active": paper["stop_active"].astype(int).values,
        }
    )
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"date": str})
    else:
        old = pd.DataFrame(columns=cols)
    known = set(old["date"])
    mismatched = []
    merged = old.merge(new, on="date", suffixes=("_old", "_new"))
    for _, row in merged.iterrows():
        if abs(row["net_return_old"] - row["net_return_new"]) > 1e-7:
            mismatched.append(row["date"])
    additions = new[~new["date"].isin(known)]
    out = pd.concat([old, additions], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return mismatched


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the Quanter daily pipeline")
    ap.add_argument("--synthetic", action="store_true", help="use fake prices (smoke test only)")
    ap.add_argument("--out", default=None, help="results.json path")
    ap.add_argument("--log", default=None, help="paper log path")
    args = ap.parse_args(argv)

    cfg = Config()
    out_path = args.out or cfg.out_path
    log_path = args.log or cfg.log_path

    prices = synthetic_prices() if args.synthetic else fetch_prices(cfg.ticker, cfg.data_start)
    res = build_results(prices, cfg, synthetic=args.synthetic)
    df, seg = res.pop("_frame"), res.pop("_segments_series")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, separators=(",", ":"))

    mismatched = update_paper_log(df, seg, log_path)
    if mismatched:
        print("WARNING: recomputed paper returns differ from the log on:", ", ".join(mismatched))
    print(f"Wrote {out_path} through {res['meta']['last_data_date']} (synthetic={args.synthetic})")


if __name__ == "__main__":
    main()
