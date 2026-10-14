from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Data
    ticker: str = "SPY"
    data_start: str = "2010-01-01"

    # Segment boundaries. Keep these fixed once set.
    # in_sample: data_start .. in_sample_end (the period you may look at while designing)
    # out_of_sample: after in_sample_end, before paper_start (looked at only to validate)
    # paper: paper_start onward (generated daily by this pipeline, never revised)
    in_sample_end: str = "2019-12-31"
    # Set this to the date you first deploy the pipeline. Never move it earlier.
    paper_start: str = "2026-10-15"

    # Baseline strategy parameters (an example, replace with your own strategy)
    target_vol: float = 0.10
    max_leverage: float = 1.0
    vol_window: int = 20
    trend_window: int = 200

    # Drawdown stop: after the strategy's own net equity falls this far from its peak,
    # go flat for dd_cooldown trading days, then re-arm from the new equity level.
    # Set dd_stop to 0 to disable.
    dd_stop: float = 0.15
    dd_cooldown: int = 21

    # Decay check: compare the latest rolling Sharpe (decay_window days) with the
    # distribution of the same statistic over the out-of-sample segment.
    decay_window: int = 126
    decay_quantile: float = 0.05

    # Frictions and conventions
    cost_bps: float = 2.0  # per unit of turnover, one way
    risk_free_rate: float = 0.0  # Sharpe ratios are computed on raw returns
    trading_days: int = 252

    # Risk metrics
    sharpe_window: int = 63
    var_window: int = 504
    var_level: float = 0.95

    # Output
    out_path: str = "site_data/results.json"
    log_path: str = "site_data/paper_log.csv"
