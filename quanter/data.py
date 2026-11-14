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


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def parse_fred_csv(text: str) -> pd.Series:
    """Parse FRED's fredgraph.csv (date, value; '.' for missing) to a float Series."""
    from io import StringIO

    df = pd.read_csv(StringIO(text), na_values=["."])
    s = df.iloc[:, 1].astype(float)
    s.index = pd.to_datetime(df.iloc[:, 0]).dt.normalize()
    return s.dropna()


def fetch_tbill_rate(series: str = "DGS3MO", timeout: int = 20) -> Optional[pd.Series]:
    """Annualised 3-month T-bill yield (decimal) from FRED, no API key needed.

    Returns None if FRED cannot be reached, so the daily job still runs and
    simply reports the excess-of-cash Sharpe as null.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(FRED_CSV.format(series=series), timeout=timeout) as r:
            text = r.read().decode("utf-8")
        return parse_fred_csv(text) / 100.0
    except Exception:
        return None


def daily_rf(rate: pd.Series, index: pd.DatetimeIndex, trading_days: int = 252) -> pd.Series:
    """Daily cash return on each trading day, from the rate known at the prior close.

    The annual yield is forward-filled onto the trading calendar, shifted one
    day (the rate for day t is the one published by t-1), and converted with
    (1 + r) ** (1 / trading_days) - 1.
    """
    r = rate.reindex(rate.index.union(index)).ffill().reindex(index).shift(1)
    return (1.0 + r) ** (1.0 / trading_days) - 1.0
