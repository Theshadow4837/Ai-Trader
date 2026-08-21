import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DATA_FILE = Path("data/SPY_features.csv")

INITIAL_CAPITAL = 10_000
TRADING_COST = 0.001

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


def create_model():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ])


def walk_forward():

    data = pd.read_csv(DATA_FILE)

    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Remove rows that don't have complete features
    data = data.dropna(
        subset=FEATURES + ["target"]
    ).reset_index(drop=True)

    # We use yearly test periods.
    first_test_year = 2020
    last_test_year = data["Date"].dt.year.max()

    results = []

    print()
    print("================================")
    print("       WALK-FORWARD TEST")
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
        y_train = train["target"]

        X_test = test[FEATURES]
        y_test = test["target"]

        model = create_model()

        model.fit(
            X_train,
            y_train
        )

        test["prob_up"] = (
            model.predict_proba(X_test)[:, 1]
        )

        test["signal"] = (
            test["prob_up"] >= 0.5
        ).astype(int)

        # Next trading day's return
        test["market_return"] = (
            test["Close"].shift(-1)
            / test["Close"]
            - 1
        )

        test["position_change"] = (
            test["signal"].diff().abs()
        )

        test["strategy_return"] = (
            test["signal"]
            * test["market_return"]
        )

        test["strategy_return"] -= (
            test["position_change"]
            * TRADING_COST
        )

        test = test.dropna().copy()

        ai_return = (
            (1 + test["strategy_return"]).prod()
            - 1
        )

        buy_hold_return = (
            (1 + test["market_return"]).prod()
            - 1
        )

        ai_sharpe = sharpe_ratio(
            test["strategy_return"]
        )

        buy_hold_sharpe = sharpe_ratio(
            test["market_return"]
        )

        results.append({
            "year": test_year,
            "ai_return": ai_return,
            "buy_hold_return": buy_hold_return,
            "ai_sharpe": ai_sharpe,
            "buy_hold_sharpe": buy_hold_sharpe,
            "trades": int(
                test["position_change"].sum()
            )
        })

        print(
            f"\n{test_year}"
        )

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
            f"Trades: "
            f"{int(test['position_change'].sum())}"
        )

    results = pd.DataFrame(results)

    print()
    print("================================")
    print("          SUMMARY")
    print("================================")

    print(
        f"\nAverage AI annual return: "
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
        f"\nTotal trades: "
        f"{results['trades'].sum()}"
    )


if __name__ == "__main__":
    walk_forward()