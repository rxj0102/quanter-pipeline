"""Decay check: is the strategy still behaving like it did when it was validated?

Two comparisons, both against the out-of-sample segment, which is the validated range:
  1. The latest rolling Sharpe against the lower tail of the same statistic over
     the out-of-sample segment.
  2. The current drawdown (measured from the peak since out-of-sample began)
     against the worst drawdown seen inside the out-of-sample segment.

Either one outside the range sets the status to review. This flags a strategy
for a human look. It does not retire anything automatically.
"""
import pandas as pd

from . import metrics as m


def _round(x, nd=6):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def decay_check(
    net: pd.Series,
    seg: pd.Series,
    window: int,
    quantile: float = 0.05,
    min_reference: int = 60,
) -> dict:
    out = {
        "status": "insufficient_data",
        "window": window,
        "quantile": quantile,
        "reference_days": 0,
        "current_sharpe": None,
        "lower_bound": None,
        "percentile": None,
        "current_drawdown": None,
        "worst_validated_drawdown": None,
        "flags": [],
    }
    if len(net) < window:
        return out

    rs = m.rolling_sharpe(net, window)
    ref = rs[seg == "out_of_sample"].dropna()
    out["reference_days"] = int(len(ref))

    # Flat for the whole window: nothing to evaluate.
    if (net.iloc[-window:] == 0).all():
        out["status"] = "flat"
        return out

    current = rs.iloc[-1]
    oos_net = net[seg == "out_of_sample"]
    if len(ref) < min_reference or pd.isna(current) or len(oos_net) < 2:
        return out

    lower = float(ref.quantile(quantile))
    percentile = float((ref <= current).mean())
    post_net = net[seg != "in_sample"]
    dd_now = float(m.drawdown(post_net).iloc[-1])
    worst = float(m.max_drawdown(oos_net))

    flags = []
    if current < lower:
        flags.append("sharpe_below_validated_range")
    if dd_now < worst:
        flags.append("drawdown_beyond_validated")

    out.update(
        {
            "status": "review" if flags else "within_range",
            "current_sharpe": _round(current),
            "lower_bound": _round(lower),
            "percentile": _round(percentile, 4),
            "current_drawdown": _round(dd_now),
            "worst_validated_drawdown": _round(worst),
            "flags": flags,
        }
    )
    return out
