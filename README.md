# Ai-Trader

Ai-Trader is a market research project centered on the frozen V28 PPO model. The current production path is a forward-only, deterministic paper-trading system for SPY LONG/FLAT decisions.

This repository is for research and simulated paper trading only. It does not connect to a brokerage account, authenticate to a broker, submit real orders, or manage real money.

## V28 Frozen Model

The active model is:

```text
models/v28/v28_seed_202_FROZEN.zip
```

Expected SHA-256:

```text
eb9c2dcd1e9cc1ac1ae85f1a4e759aae4aad57b10c9a0ce7a1320dfefd2245ad
```

V28 uses PPO with `Discrete(2)` actions and an observation space of `Box(-10.0, 10.0, (85,), float32)`.

Actions:

- `0` = FLAT
- `1` = LONG

Do not retrain, fine-tune, overwrite, or optimize this frozen model from paper results.

## Data Pipeline

Raw market data lives in:

```text
data/SPY.csv
data/QQQ.csv
data/IWM.csv
data/DIA.csv
data/VIX.csv
```

Update all five datasets with:

```bash
.venv/bin/python src/multi_data_loader.py
```

The updater uses `yfinance`, removes duplicate dates, sorts rows, validates OHLCV data, reports each dataset's latest date, and stops if the five datasets are not synchronized.

## Live Feature Pipeline

Generate the V28 live feature file with:

```bash
.venv/bin/python src/features_live_v24.py
```

This writes:

```text
data/live_features_v24.csv
```

The live feature generator validates that the final file has exactly 85 V28 features, in the same order as `data/market_features_v14.csv`, with no future-return or target columns and no NaN or infinite feature values.

## Paper Inference

Run deterministic forward-only inference with:

```bash
.venv/bin/python src/v28_paper_runner.py
```

The runner:

- verifies the frozen model SHA-256 before inference
- validates the 85-feature schema
- uses training-period normalization statistics only
- runs deterministic V28 inference
- logs new LONG/FLAT decisions once per market date
- updates the persistent simulated account
- exits cleanly on duplicate or stale data

Decision log:

```text
data/v28_paper_log.csv
```

## Persistent Paper Account

The simulated account files are:

```text
data/v28_paper_account.json
data/v28_paper_equity.csv
data/v28_paper_trades.csv
```

The account tracks simulated cash, position, position size, entry price, equity, realized and unrealized P/L, cumulative return, drawdown, trade counts, win rate, time in market, transaction costs, and last processed market date.

Idempotency is required: running the paper runner twice on the same market date must not create duplicate predictions, duplicate trades, duplicate equity rows, or duplicate transaction costs.

## Performance Report

Read the current paper account status with:

```bash
.venv/bin/python src/v28_paper_report.py
```

The report is read-only. It summarizes current simulated equity, cumulative return, annualized return, max drawdown, Sharpe ratio, trades, win rate, average trade return, exposure, turnover, transaction costs, longest drawdown, best/worst trade, daily returns, rolling returns, current position, and latest model action.

All reported performance is `SIMULATED / PAPER`. It is not real investment performance and must not be treated as proof that the model can make money in live trading. The paper account currently reinvests the full simulated portfolio on each LONG entry and was bootstrapped from historical V28 decision logs, so large account values mostly reflect full-portfolio compounding under simplified execution assumptions.

## Daily Health Check

Run the full unattended production-hardening check with:

```bash
scripts/v28_daily_check.sh
```

The check verifies the frozen model file and SHA-256, updates market data, generates live features, validates the 85-feature schema, runs the paper trader, confirms a second same-date run leaves account/log/equity/trade files unchanged, generates the performance report, and runs the unit tests. Any critical failure exits with a non-zero status.

## Daily Workflow

```bash
.venv/bin/python src/multi_data_loader.py
.venv/bin/python src/features_live_v24.py
.venv/bin/python src/v28_paper_runner.py
.venv/bin/python src/v28_paper_report.py
```

If there is no new market date, the runner prints `NO NEW MARKET DATA` and exits without changing account state.

If the live feature dataset is older than the latest processed paper date, the runner raises `STALE MARKET DATA` and stops. The system does not invent rows for weekends, holidays, missing sessions, or delayed data.

## Tests

Run the V28 paper-trading tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover frozen model hash verification, feature schema integrity, future-column rejection, NaN detection, training-only normalization behavior, stale and backwards data handling, duplicate-run protection, position transitions, transaction costs, persistent account recovery, equity and drawdown calculation, trade logging, market data cleaning, paper-only execution checks, and frozen model file immutability during validation.
# Ai-Trader
