from typing import Optional

import numpy as np
import pandas as pd


def drop_incomplete_bar(
    close: pd.Series,
    now: Optional[pd.Timestamp] = None,
    close_hour: int = 16,
    close_minute: int = 30,
) -> pd.Series:
    """Drop today's bar if the US market has not closed yet.

    During market hours Yahoo returns a partial bar for today. Treating it as a
    final close would put a wrong number into the results, and into the
    append-only paper log, permanently.
    """
    if close.empty:
        return close
    now = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    cutoff = now.normalize().replace(hour=close_hour, minute=close_minute)
    if close.index[-1].date() == now.date() and now < cutoff:
        return close.iloc[:-1]
    return close


def fetch_prices(ticker: str, start: str) -> pd.Series:
    """Download adjusted daily closes. Returns a Series indexed by date."""
    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No price data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.name = ticker
    return drop_incomplete_bar(close)


def synthetic_prices(start: str = "2010-01-01", end: str = "2026-10-20", seed: int = 7) -> pd.Series:
    """Fake prices for tests and smoke runs only. Never used for published numbers."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        stay = 0.985 if regime[i - 1] == 0 else 0.95
        regime[i] = regime[i - 1] if rng.random() < stay else 1 - regime[i - 1]
    mu = np.where(regime == 0, 0.0005, -0.0004)
    sigma = np.where(regime == 0, 0.007, 0.02)
    rets = rng.normal(mu, sigma)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=idx, name="SYNTH")
