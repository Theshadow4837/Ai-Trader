import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor


DATA_FILE = Path("data/market_features_v10.csv")

INITIAL_CAPITAL = 10_000.0

# Approximate one-way trading cost.
TRANSACTION_COST = 0.0005

THRESHOLDS = [
    0.0000,
    0.0010,
    0.0020,
    0.0030,
    0.0050,
]


def sharpe_ratio(returns):

    if returns.std() == 0:
        return 0.0

    return (
        np.sqrt(252)
        * returns.mean()
        / returns.std()
    )


def max_drawdown(equity):

    peak = equity.cummax()

    drawdown = (
        equity / peak
        - 1
    )

    return drawdown.min()


def run_backtest(data, threshold):

    capital = INITIAL_CAPITAL

    previous_position = 0

    equity_curve = []

    trades = 0

    exposure_days = 0

    daily_returns = []

    for _, row in data.iterrows():

        prediction = row["prediction"]

        actual_return = row[
            "next_day_return"
        ]

        # Long only.
        if prediction > threshold:
            position = 1
        else:
            position = 0

        # Count trade when position changes.
        if position != previous_position:

            if previous_position != 0:
                capital *= (
                    1 - TRANSACTION_COST
                )

            if position != 0:
                capital *= (
                    1 - TRANSACTION_COST
                )

            trades += 1

        if position == 1:
            exposure_days += 1

        strategy_return = (
            actual_return
            * position
        )

        capital *= (
            1 + strategy_return
        )

        daily_returns.append(
            strategy_return
        )

        equity_curve.append(
            capital
        )

        previous_position = position

    returns = pd.Series(
        daily_returns
    )

    equity = pd.Series(
        equity_curve
    )

    total_return = (
        capital / INITIAL_CAPITAL
        - 1
    )

    years = len(data) / 252

    annualized_return = (
        (capital / INITIAL_CAPITAL)
        ** (1 / years)
        - 1
    )

    return {
        "threshold": threshold,
        "final_value": capital,
        "return": total_return,
        "annualized_return":
            annualized_return,
        "max_drawdown":
            max_drawdown(equity),
        "sharpe":
            sharpe_ratio(returns),
        "trades": trades,
        "exposure":
            exposure_days / len(data)
    }


def main():

    data = pd.read_csv(
        DATA_FILE
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .dropna()
        .reset_index(drop=True)
    )

    FEATURES = [
        column
        for column in data.columns
        if column not in [
            "Date",
            "next_day_return"
        ]
    ]

    all_predictions = []

    print()
    print("================================")
    print("     GENERATING WALK-FORWARD")
    print("          PREDICTIONS")
    print("================================")

    for test_year in range(
        2020,
        2027
    ):

        train = data[
            data["Date"].dt.year < test_year
        ]

        test = data[
            data["Date"].dt.year == test_year
        ].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            loss="huber",
            random_state=42
        )

        print(
            f"\nTraining through "
            f"{test_year - 1}..."
        )

        model.fit(
            train[FEATURES],
            train["next_day_return"]
        )

        test["prediction"] = model.predict(
            test[FEATURES]
        )

        all_predictions.append(
            test[
                [
                    "Date",
                    "next_day_return",
                    "prediction"
                ]
            ]
        )

    results = pd.concat(
        all_predictions,
        ignore_index=True
    )

    print()
    print("================================")
    print("        V12 BACKTEST")
    print("================================")

    print(
        f"Testing period: "
        f"{results['Date'].min().date()} "
        f"→ "
        f"{results['Date'].max().date()}"
    )

    print(
        f"Initial capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST * 100:.02f}%"
    )

    output = []

    for threshold in THRESHOLDS:

        result = run_backtest(
            results,
            threshold
        )

        output.append(result)

    comparison = pd.DataFrame(
        output
    )

    comparison[
        "threshold"
    ] *= 100

    comparison[
        "return"
    ] *= 100

    comparison[
        "annualized_return"
    ] *= 100

    comparison[
        "max_drawdown"
    ] *= 100

    comparison[
        "exposure"
    ] *= 100

    print()
    print(
        comparison.to_string(
            index=False,
            formatters={
                "threshold":
                    "{:.2f}%".format,
                "final_value":
                    "${:,.2f}".format,
                "return":
                    "{:.2f}%".format,
                "annualized_return":
                    "{:.2f}%".format,
                "max_drawdown":
                    "{:.2f}%".format,
                "sharpe":
                    "{:.3f}".format,
                "exposure":
                    "{:.1f}%".format,
            }
        )
    )

    # Buy and hold.
    buy_hold_returns = results[
        "next_day_return"
    ]

    bh_equity = (
        INITIAL_CAPITAL
        * (1 + buy_hold_returns).cumprod()
    )

    bh_final = bh_equity.iloc[-1]

    bh_return = (
        bh_final
        / INITIAL_CAPITAL
        - 1
    )

    bh_years = (
        len(results) / 252
    )

    bh_annualized = (
        (bh_final / INITIAL_CAPITAL)
        ** (1 / bh_years)
        - 1
    )

    print()
    print("================================")
    print("          BUY & HOLD")
    print("================================")

    print(
        f"Final value: "
        f"${bh_final:,.2f}"
    )

    print(
        f"Return: "
        f"{bh_return * 100:.2f}%"
    )

    print(
        f"Annualized return: "
        f"{bh_annualized * 100:.2f}%"
    )

    print(
        f"Max drawdown: "
        f"{max_drawdown(bh_equity) * 100:.2f}%"
    )

    print(
        f"Sharpe: "
        f"{sharpe_ratio(buy_hold_returns):.3f}"
    )


if __name__ == "__main__":
    main()