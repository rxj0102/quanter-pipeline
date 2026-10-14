import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    return float(drawdown(returns).min())


def sharpe(returns: pd.Series, trading_days: int = 252, rf: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - rf / trading_days
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(trading_days))


def rolling_sharpe(returns: pd.Series, window: int, trading_days: int = 252, rf: float = 0.0) -> pd.Series:
    excess = returns - rf / trading_days
    mean = excess.rolling(window).mean()
    sd = excess.rolling(window).std()
    return (mean / sd * np.sqrt(trading_days)).replace([np.inf, -np.inf], np.nan)


def hist_var_cvar(returns: pd.Series, window: int, level: float = 0.95):
    """Historical 1-day VaR and CVaR over the trailing window, reported as positive losses."""
    tail = returns.dropna().iloc[-window:]
    if len(tail) < 30:
        return float("nan"), float("nan")
    q = tail.quantile(1.0 - level)
    var = -float(q)
    worst = tail[tail <= q]
    cvar = -float(worst.mean())
    return var, cvar


def ann_return(returns: pd.Series, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return float("nan")
    growth = float(equity_curve(returns).iloc[-1])
    return growth ** (trading_days / len(returns)) - 1.0


def ann_vol(returns: pd.Series, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std() * np.sqrt(trading_days))
