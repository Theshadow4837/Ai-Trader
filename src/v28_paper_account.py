"""Persistent, local-only V28 paper portfolio accounting.

This module deliberately contains no broker, network, or order-execution code.
Orders are simulated at the supplied daily SPY close, after the model action has
been recorded for that date.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


INITIAL_CAPITAL = 10.0
TRANSACTION_COST_RATE = 0.0005


@dataclass
class PaperAccount:
    starting_capital: float = INITIAL_CAPITAL
    cash: float = INITIAL_CAPITAL
    position: int = 0
    position_size: float = 0.0
    entry_date: str | None = None
    entry_price: float | None = None
    entry_cost: float = 0.0
    current_equity: float = INITIAL_CAPITAL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    cumulative_return: float = 0.0
    peak_equity: float = INITIAL_CAPITAL
    maximum_drawdown: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_days: int = 0
    days_in_market: int = 0
    transaction_costs: float = 0.0
    last_processed_date: str | None = None
    last_action: int | None = None
    last_market_price: float | None = None
    last_timestamp: str | None = None

    @property
    def win_rate(self) -> float:
        closed = self.winning_trades + self.losing_trades
        return self.winning_trades / closed if closed else 0.0

    @property
    def time_in_market(self) -> float:
        return self.days_in_market / self.total_days if self.total_days else 0.0

    def to_dict(self) -> dict:
        values = asdict(self)
        values["win_rate"] = self.win_rate
        values["time_in_market"] = self.time_in_market
        values["transaction_cost_rate"] = TRANSACTION_COST_RATE
        return values


def load_account(path: Path) -> PaperAccount:
    if not path.exists():
        return PaperAccount()
    with path.open() as handle:
        raw = json.load(handle)
    allowed = {field.name for field in PaperAccount.__dataclass_fields__.values()}
    return PaperAccount(**{key: value for key, value in raw.items() if key in allowed})


def save_account(account: PaperAccount, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(account.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


class PaperPortfolio:
    def __init__(self, account_path: Path, equity_path: Path, trades_path: Path):
        self.account_path = account_path
        self.equity_path = equity_path
        self.trades_path = trades_path
        self.account = load_account(account_path)

    def process(self, date, action: int, price: float, timestamp: str | None = None) -> dict:
        """Process one date exactly once and return its daily equity record."""
        day = pd.Timestamp(date).strftime("%Y-%m-%d")
        if action not in (0, 1):
            raise ValueError(f"V28 action must be 0 (FLAT) or 1 (LONG), got {action}")
        if pd.isna(price) or price <= 0:
            raise ValueError(f"Market price must be positive for {day}")
        if self.account.last_processed_date:
            previous = pd.Timestamp(self.account.last_processed_date)
            current = pd.Timestamp(day)
            if current < previous:
                raise RuntimeError("Market data has moved backwards relative to account state.")
            if current == previous:
                raise RuntimeError(f"Date {day} is already processed by the paper account.")

        account = self.account
        transaction_cost = 0.0
        trade = None
        prior_position = account.position

        if action != prior_position:
            if action == 1:
                # Invest all available cash, including the one-way simulated cost.
                size = account.cash / (price * (1.0 + TRANSACTION_COST_RATE))
                notional = size * price
                transaction_cost = notional * TRANSACTION_COST_RATE
                account.cash -= notional + transaction_cost
                account.position = 1
                account.position_size = size
                account.entry_date = day
                account.entry_price = price
                account.entry_cost = transaction_cost
                trade = {"date": day, "side": "BUY", "price": price, "position_size": size,
                         "notional": notional, "transaction_cost": transaction_cost,
                         "realized_pnl": 0.0, "entry_date": day}
            else:
                notional = account.position_size * price
                transaction_cost = notional * TRANSACTION_COST_RATE
                proceeds = notional - transaction_cost
                entry_total = account.position_size * float(account.entry_price) + account.entry_cost
                pnl = proceeds - entry_total
                account.cash += proceeds
                account.realized_pnl += pnl
                if pnl > 0:
                    account.winning_trades += 1
                elif pnl < 0:
                    account.losing_trades += 1
                trade = {"date": day, "side": "SELL", "price": price, "position_size": account.position_size,
                         "notional": notional, "transaction_cost": transaction_cost,
                         "realized_pnl": pnl, "entry_date": account.entry_date}
                account.position = 0
                account.position_size = 0.0
                account.entry_date = None
                account.entry_price = None
                account.entry_cost = 0.0
            account.trade_count += 1
            account.transaction_costs += transaction_cost

        market_value = account.position_size * price
        account.current_equity = account.cash + market_value
        account.unrealized_pnl = (market_value - account.position_size * float(account.entry_price) - account.entry_cost
                                  if account.position else 0.0)
        account.cumulative_return = account.current_equity / account.starting_capital - 1.0
        account.peak_equity = max(account.peak_equity, account.current_equity)
        account.maximum_drawdown = min(account.maximum_drawdown,
                                       account.current_equity / account.peak_equity - 1.0)
        account.total_days += 1
        account.days_in_market += int(account.position == 1)
        account.last_processed_date = day
        account.last_action = action
        account.last_market_price = price
        account.last_timestamp = timestamp or day

        row = {"date": day, "timestamp": account.last_timestamp, "model_action": action,
               "market_price": price, "position": account.position, "position_size": account.position_size,
               "cash": account.cash, "entry_date": account.entry_date, "entry_price": account.entry_price,
               "current_equity": account.current_equity, "realized_pnl": account.realized_pnl,
               "unrealized_pnl": account.unrealized_pnl, "cumulative_return": account.cumulative_return,
               "maximum_drawdown": account.maximum_drawdown, "trade_count": account.trade_count,
               "winning_trades": account.winning_trades, "losing_trades": account.losing_trades,
               "win_rate": account.win_rate, "time_in_market": account.time_in_market,
               "transaction_cost": transaction_cost, "transaction_costs": account.transaction_costs}
        self._append(self.equity_path, row)
        if trade:
            self._append(self.trades_path, trade)
        save_account(account, self.account_path)
        return row

    @staticmethod
    def _append(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
