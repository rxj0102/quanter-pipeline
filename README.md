# Quanter pipeline

A daily job that runs one strategy, applies stated costs, labels every day as in-sample, out-of-sample or paper trading, and writes a results file for the Quanter site. Research and paper trading only.

## Run it

```
pip install -r requirements.txt
pytest -q
python -m quanter.pipeline              # real prices from Yahoo Finance
python -m quanter.pipeline --synthetic --out /tmp/r.json --log /tmp/log.csv   # smoke test, fake prices
```

Output goes to `site_data/results.json` and `site_data/paper_log.csv`.

## The baseline strategy is a placeholder

`quanter/strategy.py` holds a trend-filtered, volatility-targeted position in one ticker. It exists so the pipeline runs end to end. Replace `compute_signals` with your own research strategy, and keep one rule: the position decided at the close of day t may only use data available at the close of day t, and is earned on day t+1. `test_no_lookahead_in_positions` checks this for whatever you plug in.

## How honesty is enforced

- **Segments are fixed in `config.py`.** In-sample ends at `in_sample_end`. Paper trading starts at `paper_start`, which you set to the day you first deploy and never move earlier.
- **Paper log is append-only.** `paper_log.csv` gets new rows daily and existing rows are never rewritten. Because the GitHub Action commits it every day, git history is the timestamp proof. If a recomputation ever disagrees with a logged row, the run prints a warning.
- **Costs are in the first run.** `cost_bps` is charged on turnover in every segment.
- **Synthetic runs are flagged.** `meta.synthetic` is true in any file made from fake prices, so one can never be mistaken for real results.

## Drawdown stop

The strategy watches its own net equity. When it falls `dd_stop` (default 15 percent) from its peak, the position goes to zero for the next `dd_cooldown` trading days (default 21). After the cooldown it re-enters and measures drawdown from the new equity level. The stop only uses information through the close of the day it triggers and acts on the following day. Set `dd_stop = 0` in `config.py` to switch it off.

A stop is a trade-off, not free protection. It cuts exposure in a slide, and it also locks in losses and misses rebounds. On the synthetic test series it fired three times and did not improve the worst drawdown. Judge it on real out-of-sample results before keeping it.

## Decay check

Every run compares the strategy with its own validated range, the out-of-sample segment:

- The latest `decay_window`-day rolling Sharpe against the lower tail (`decay_quantile`, default 5th percentile) of the same statistic over the out-of-sample segment.
- The current drawdown, measured from the peak since out-of-sample began, against the worst drawdown inside the out-of-sample segment.

Status is `within_range`, `review`, `flat` (no positions during the whole window), or `insufficient_data`. A `review` status flags the strategy for a human look. It never retires anything automatically.

## Deploy

1. Push this folder to a GitHub repo.
2. Set `paper_start` in `quanter/config.py` to your deploy date and commit.
3. In the repo, run the daily-refresh workflow once from the Actions tab, then it runs on weekdays.
4. The site reads `site_data/results.json` from the repo (raw GitHub URL or a copy at build time).

## results.json

- `meta`: generated_at, last_data_date, synthetic flag, assumptions (the full config)
- `summary`: rolling_sharpe (63 days), max_drawdown_all, max_drawdown_paper, var_95_1d and cvar_95_1d (positive numbers are losses), regime, paper_days, drawdown_stop (threshold, cooldown, active, current_drawdown, trigger_count, days_stopped), decay (status, current_sharpe, lower_bound, percentile, flags and the drawdown comparison)
- `segments`: days, ann_return, ann_vol, sharpe, max_drawdown, start and end for in_sample, out_of_sample and paper
- `series`: parallel arrays of dates, segment, cum_return, drawdown, rolling_sharpe, stop_active (1 on days the stop forced the position to zero)

## Known limits

- Yahoo Finance data is free and convenient but can have gaps and revisions. Fine for a research site, not for production.
- VaR and CVaR are historical over the trailing 504 days of strategy returns.
- A paper-trading period of a few months says very little about a strategy. The site should say so.
