import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


DATA_FILE = Path("data/SPY_features.csv")

TRADING_COST = 0.001
HOLDING_DAYS = 5
ENTRY_THRESHOLD = 0.005

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


def sharpe_ratio(returns):

    if len(returns) < 2 or returns.std() == 0:
        return 0

    return (
        returns.mean()
        / returns.std()
        * np.sqrt(252 / HOLDING_DAYS)
    )


def create_model():

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )


def walk_forward():

    data = pd.read_csv(DATA_FILE)

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    data = data.dropna(
        subset=FEATURES
    ).reset_index(drop=True)

    # Future 5-trading-day return
    data["future_return"] = (
        data["Close"].shift(-HOLDING_DAYS)
        / data["Close"]
        - 1
    )

    data = data.dropna(
        subset=["future_return"]
    ).reset_index(drop=True)

    first_test_year = 2020
    last_test_year = data["Date"].dt.year.max()

    results = []

    print()
    print("================================")
    print("       V4.1 WALK-FORWARD")
    print("   NON-OVERLAPPING 5-DAY TEST")
    print("================================")

    for test_year in range(
        first_test_year,
        last_test_year + 1
    ):

        train = data[
            data["Date"].dt.year < test_year
        ].copy()

        test = data[
            data["Date"].dt.year == test_year
        ].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        X_train = train[FEATURES]
        y_train = train["future_return"]

        model = create_model()

        print(
            f"\nTraining model for {test_year}..."
        )

        model.fit(
            X_train,
            y_train
        )

        # Walk through the test year in
        # non-overlapping 5-day blocks.
        trades = []

        i = 0

        while i + HOLDING_DAYS < len(test):

            entry = test.iloc[i]

            X_entry = entry[FEATURES].to_frame().T

            predicted_return = model.predict(
                X_entry
            )[0]

            entry_price = entry["Close"]

            exit_row = test.iloc[
                i + HOLDING_DAYS
            ]

            exit_price = exit_row["Close"]

            actual_return = (
                exit_price / entry_price - 1
            )

            if predicted_return > ENTRY_THRESHOLD:

                strategy_return = (
                    actual_return
                    - TRADING_COST
                )

                trades.append({
                    "entry_date": entry["Date"],
                    "exit_date": exit_row["Date"],
                    "predicted_return": predicted_return,
                    "actual_return": actual_return,
                    "strategy_return": strategy_return
                })

            else:

                trades.append({
                    "entry_date": entry["Date"],
                    "exit_date": exit_row["Date"],
                    "predicted_return": predicted_return,
                    "actual_return": 0.0,
                    "strategy_return": 0.0
                })

            i += HOLDING_DAYS

        trades = pd.DataFrame(trades)

        if len(trades) == 0:
            continue

        ai_return = (
            (1 + trades["strategy_return"]).prod()
            - 1
        )

        # Buy & hold over the same 5-day blocks
        market_returns = []

        i = 0

        while i + HOLDING_DAYS < len(test):

            entry_price = test.iloc[i]["Close"]

            exit_price = test.iloc[
                i + HOLDING_DAYS
            ]["Close"]

            market_returns.append(
                exit_price / entry_price - 1
            )

            i += HOLDING_DAYS

        market_returns = pd.Series(
            market_returns
        )

        buy_hold_return = (
            (1 + market_returns).prod()
            - 1
        )

        ai_sharpe = sharpe_ratio(
            trades["strategy_return"]
        )

        buy_hold_sharpe = sharpe_ratio(
            market_returns
        )

        active_trades = (
            trades["strategy_return"] != 0
        ).sum()

        exposure = (
            active_trades
            / len(trades)
            * 100
        )

        results.append({
            "year": test_year,
            "ai_return": ai_return,
            "buy_hold_return": buy_hold_return,
            "ai_sharpe": ai_sharpe,
            "buy_hold_sharpe": buy_hold_sharpe,
            "trades": active_trades,
            "exposure": exposure
        })

        print(
            f"AI return: "
            f"{ai_return * 100:.2f}%"
        )

        print(
            f"Buy & hold: "
            f"{buy_hold_return * 100:.2f}%"
        )

        print(
            f"AI Sharpe: "
            f"{ai_sharpe:.3f}"
        )

        print(
            f"Buy & hold Sharpe: "
            f"{buy_hold_sharpe:.3f}"
        )

        print(
            f"AI trades: "
            f"{active_trades}"
        )

        print(
            f"AI exposure: "
            f"{exposure:.1f}%"
        )

    results = pd.DataFrame(results)

    print()
    print("================================")
    print("             SUMMARY")
    print("================================")

    print(
        f"\nAverage AI return: "
        f"{results['ai_return'].mean() * 100:.2f}%"
    )

    print(
        f"Average buy & hold return: "
        f"{results['buy_hold_return'].mean() * 100:.2f}%"
    )

    print(
        f"\nAverage AI Sharpe: "
        f"{results['ai_sharpe'].mean():.3f}"
    )

    print(
        f"Average buy & hold Sharpe: "
        f"{results['buy_hold_sharpe'].mean():.3f}"
    )

    print(
        f"\nTotal AI trades: "
        f"{results['trades'].sum()}"
    )

    print(
        f"Average AI exposure: "
        f"{results['exposure'].mean():.1f}%"
    )


if __name__ == "__main__":
    walk_forward()