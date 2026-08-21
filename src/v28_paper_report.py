"""Read-only performance report for the frozen V28 paper account."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ACCOUNT_FILE = Path("data/v28_paper_account.json")
EQUITY_FILE = Path("data/v28_paper_equity.csv")
TRADES_FILE = Path("data/v28_paper_trades.csv")
LOG_FILE = Path("data/v28_paper_log.csv")
TRADING_DAYS_PER_YEAR = 252


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_account() -> dict:
    if not ACCOUNT_FILE.exists():
        raise FileNotFoundError(f"Paper account not found: {ACCOUNT_FILE}")
    with ACCOUNT_FILE.open() as handle:
        return json.load(handle)


def daily_returns(equity: pd.DataFrame) -> pd.Series:
    return equity["current_equity"].pct_change().dropna()


def annualized_return(equity: pd.DataFrame) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity["current_equity"].iloc[-1] / equity["current_equity"].iloc[0] - 1.0
    years = (len(equity) - 1) / TRADING_DAYS_PER_YEAR
    return (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0


def sharpe_ratio(equity: pd.DataFrame) -> float:
    returns = daily_returns(equity)
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float((returns.mean() / std) * (TRADING_DAYS_PER_YEAR ** 0.5))


def longest_drawdown_days(equity: pd.DataFrame) -> int:
    if equity.empty:
        return 0
    values = equity["current_equity"]
    below_peak = values < values.cummax()
    longest = 0
    current = 0
    for active in below_peak:
        if active:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def closed_trade_returns(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    returns = []
    open_entry_cost = None
    for trade in trades.sort_values("date").itertuples(index=False):
        side = str(trade.side).upper()
        notional = float(trade.notional)
        cost = float(trade.transaction_cost)
        if side == "BUY":
            open_entry_cost = notional + cost
        elif side == "SELL" and open_entry_cost and open_entry_cost > 0:
            returns.append(float(trade.realized_pnl) / open_entry_cost)
            open_entry_cost = None
    return pd.Series(returns, dtype=float)


def turnover(equity: pd.DataFrame, trades: pd.DataFrame) -> float:
    if equity.empty or trades.empty:
        return 0.0
    average_equity = equity["current_equity"].mean()
    if average_equity <= 0:
        return 0.0
    return float(trades["notional"].abs().sum() / average_equity)


def print_rolling_performance(equity: pd.DataFrame) -> None:
    returns = daily_returns(equity)
    print()
    print("ROLLING PERFORMANCE")
    if returns.empty:
        print("No daily returns available yet.")
        return
    for window in (5, 20, 60, 252):
        if len(equity) <= window:
            continue
        value = equity["current_equity"].iloc[-1] / equity["current_equity"].iloc[-window - 1] - 1.0
        print(f"{window:>3} trading days: {pct(value)}")


def main() -> None:
    account = load_account()
    equity = pd.read_csv(EQUITY_FILE) if EQUITY_FILE.exists() else pd.DataFrame()
    trades = pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()
    log = pd.read_csv(LOG_FILE) if LOG_FILE.exists() else pd.DataFrame()

    print()
    print("=" * 60)
    print("V28 PAPER PERFORMANCE REPORT")
    print("=" * 60)
    print("SIMULATED / PAPER TRADING ONLY")
    print("NO REAL ORDERS")
    print("NO MODEL TRAINING")
    print("NOT REAL INVESTMENT PERFORMANCE")

    print()
    print("ACCOUNT")
    print(f"Last processed date: {account.get('last_processed_date')}")
    print(f"Current equity:      ${account.get('current_equity', 0.0):,.2f}")
    print(f"Cumulative return:   {pct(account.get('cumulative_return', 0.0))}")
    print(f"Annualized return:   {pct(annualized_return(equity)) if not equity.empty else 'n/a'}")
    print(f"Max drawdown:        {pct(account.get('maximum_drawdown', 0.0))}")
    print(f"Sharpe ratio:        {sharpe_ratio(equity):.3f}" if not equity.empty else "Sharpe ratio:        n/a")
    print(f"Realized P/L:        ${account.get('realized_pnl', 0.0):,.2f}")
    print(f"Unrealized P/L:      ${account.get('unrealized_pnl', 0.0):,.2f}")
    print(f"Transaction costs:   ${account.get('transaction_costs', 0.0):,.2f}")
    print(f"Longest drawdown:    {longest_drawdown_days(equity)} trading days" if not equity.empty else "Longest drawdown:    n/a")

    print()
    print("POSITION")
    print(f"Latest action:       {account.get('last_action')}")
    print(f"Current position:    {account.get('position')}")
    print(f"Position size:       {account.get('position_size', 0.0):,.6f}")
    print(f"Entry date:          {account.get('entry_date')}")
    print(f"Entry price:         {account.get('entry_price')}")
    print(f"Last market price:   {account.get('last_market_price')}")

    print()
    print("ACTIVITY")
    print(f"Predictions logged:  {len(log)}")
    print(f"Equity rows:         {len(equity)}")
    print(f"Trade rows:          {len(trades)}")
    print(f"Trade count:         {account.get('trade_count', 0)}")
    print(f"Winning trades:      {account.get('winning_trades', 0)}")
    print(f"Losing trades:       {account.get('losing_trades', 0)}")
    print(f"Win rate:            {pct(account.get('win_rate', 0.0))}")
    print(f"Exposure:            {pct(account.get('time_in_market', 0.0))}")
    print(f"Turnover:            {turnover(equity, trades):.2f}x" if not equity.empty else "Turnover:            n/a")

    trade_returns = closed_trade_returns(trades)
    print(f"Average trade return: {pct(trade_returns.mean()) if len(trade_returns) else 'n/a'}")
    print(f"Best trade:          {pct(trade_returns.max()) if len(trade_returns) else 'n/a'}")
    print(f"Worst trade:         {pct(trade_returns.min()) if len(trade_returns) else 'n/a'}")

    if not equity.empty:
        returns = daily_returns(equity)
        print()
        print("DAILY RETURNS")
        print(f"Latest daily return: {pct(returns.iloc[-1]) if len(returns) else 'n/a'}")
        print(f"Average daily return: {pct(returns.mean()) if len(returns) else 'n/a'}")
        print_rolling_performance(equity)

    print()
    print("SIMULATED EQUITY DIAGNOSTIC")
    print("The large account value is not evidence of live tradability.")
    print("This account compounds by reinvesting the full simulated portfolio on each LONG entry.")
    print("The current ledger was also bootstrapped from historical V28 paper decisions back to 2015.")
    print("No duplicate rows or repeated same-date costs are present in the current account state.")
    print("Execution assumes fills at the supplied daily SPY close with a fixed transaction-cost model.")


if __name__ == "__main__":
    main()
