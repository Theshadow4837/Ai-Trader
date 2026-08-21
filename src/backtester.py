import pandas as pd
import numpy as np
import joblib

from pathlib import Path


DATA_FILE = Path("data/SPY_features.csv")
MODEL_FILE = Path("models/v2_logistic_model.joblib")

INITIAL_CAPITAL = 10_000
TRADING_COST = 0.001  # 0.10% per position change

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "distance_sma10",
    "distance_sma50",
    "volatility_10d",
    "volatility_20d",
    "volume_change",
    "volume_ratio",
    "daily_range",
]


def max_drawdown(equity):
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min()


def sharpe_ratio(returns):
    if returns.std() == 0:
        return 0

    return (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
    )


def backtest():

    data = pd.read_csv(DATA_FILE)

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Same chronological split used during training
    split_index = int(len(data) * 0.8)

    test = data.iloc[split_index:].copy()

    model = joblib.load(MODEL_FILE)

    X_test = test[FEATURES]

    # Probability that next day is UP
    probabilities = model.predict_proba(X_test)[:, 1]

    test["prob_up"] = probabilities

    # Trading signal
    test["signal"] = (
        test["prob_up"] >= 0.5
    ).astype(int)

    # Next-day market return
    test["market_return"] = (
        test["Close"].shift(-1)
        / test["Close"]
        - 1
    )

    # Position change
    test["position_change"] = (
        test["signal"].diff().abs()
    )

    # Strategy return before costs
    test["gross_return"] = (
        test["signal"]
        * test["market_return"]
    )

    # Transaction costs
    test["transaction_cost"] = (
        test["position_change"]
        * TRADING_COST
    )

    # Net return
    test["strategy_return"] = (
        test["gross_return"]
        - test["transaction_cost"]
    )

    test = test.dropna().copy()

    # Equity curves
    test["strategy_equity"] = (
        INITIAL_CAPITAL
        * (1 + test["strategy_return"])
        .cumprod()
    )

    test["buy_hold_equity"] = (
        INITIAL_CAPITAL
        * (1 + test["market_return"])
        .cumprod()
    )

    # Returns
    strategy_return = (
        test["strategy_equity"].iloc[-1]
        / INITIAL_CAPITAL
        - 1
    )

    buy_hold_return = (
        test["buy_hold_equity"].iloc[-1]
        / INITIAL_CAPITAL
        - 1
    )

    # Risk
    strategy_drawdown = max_drawdown(
        test["strategy_equity"]
    )

    buy_hold_drawdown = max_drawdown(
        test["buy_hold_equity"]
    )

    strategy_sharpe = sharpe_ratio(
        test["strategy_return"]
    )

    buy_hold_sharpe = sharpe_ratio(
        test["market_return"]
    )

    # Trading statistics
    trades = int(
        test["position_change"].sum()
    )

    days_in_market = int(
        test["signal"].sum()
    )

    market_exposure = (
        days_in_market
        / len(test)
        * 100
    )

    # Print results
    print()
    print("================================")
    print("         V2.1 BACKTEST")
    print("================================")

    print(
        f"\nStarting capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print("\n--- AI STRATEGY ---")

    print(
        f"Final value: "
        f"${test['strategy_equity'].iloc[-1]:,.2f}"
    )

    print(
        f"Return: "
        f"{strategy_return * 100:.2f}%"
    )

    print(
        f"Max drawdown: "
        f"{strategy_drawdown * 100:.2f}%"
    )

    print(
        f"Sharpe ratio: "
        f"{strategy_sharpe:.3f}"
    )

    print(
        f"Trades: {trades}"
    )

    print(
        f"Market exposure: "
        f"{market_exposure:.1f}%"
    )

    print("\n--- BUY & HOLD ---")

    print(
        f"Final value: "
        f"${test['buy_hold_equity'].iloc[-1]:,.2f}"
    )

    print(
        f"Return: "
        f"{buy_hold_return * 100:.2f}%"
    )

    print(
        f"Max drawdown: "
        f"{buy_hold_drawdown * 100:.2f}%"
    )

    print(
        f"Sharpe ratio: "
        f"{buy_hold_sharpe:.3f}"
    )

    print("\n--- COMPARISON ---")

    print(
        f"Return difference: "
        f"{(strategy_return - buy_hold_return) * 100:.2f}%"
    )

    print(
        f"Drawdown difference: "
        f"{(strategy_drawdown - buy_hold_drawdown) * 100:.2f}%"
    )

    print(
        f"Sharpe difference: "
        f"{strategy_sharpe - buy_hold_sharpe:.3f}"
    )

    print()


if __name__ == "__main__":
    backtest()