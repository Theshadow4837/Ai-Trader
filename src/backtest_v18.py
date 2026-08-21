import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor


DATA_FILE = Path("data/market_features_v14.csv")
SPY_FILE = Path("data/SPY.csv")

INITIAL_CAPITAL = 10_000.0

HOLDING_PERIODS = [1, 3, 5, 10]

TRANSACTION_COSTS = [
    0.0000,
    0.0005,
    0.0010,
    0.0020,
]

THRESHOLD = 0.0020


def max_drawdown(equity):
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return drawdown.min()


def sharpe_ratio(returns):
    if returns.std() == 0:
        return 0.0

    return (
        np.sqrt(252)
        * returns.mean()
        / returns.std()
    )


def generate_predictions(data, features):

    predictions = []

    for test_year in range(2020, 2027):

        train = data[
            data["Date"].dt.year < test_year
        ]

        test = data[
            data["Date"].dt.year == test_year
        ].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        print(
            f"Training through {test_year - 1}..."
        )

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            loss="huber",
            random_state=42
        )

        model.fit(
            train[features],
            train["future_5d_return"]
        )

        test["prediction"] = model.predict(
            test[features]
        )

        predictions.append(
            test[
                [
                    "Date",
                    "prediction",
                    "next_day_return"
                ]
            ]
        )

    return pd.concat(
        predictions,
        ignore_index=True
    )


def run_strategy(
    dataset,
    holding_days,
    transaction_cost
):

    dataset = dataset.reset_index(drop=True)

    position = np.zeros(len(dataset))

    i = 0

    while i < len(dataset):

        prediction = dataset.loc[
            i,
            "prediction"
        ]

        # Only take positive predictions above
        # our already-selected V16 threshold.
        if prediction >= THRESHOLD:

            start = i + 1

            end = min(
                start + holding_days,
                len(dataset)
            )

            position[start:end] = 1.0

            i = end

        else:
            i += 1

    position = pd.Series(
        position,
        index=dataset.index
    )

    daily_return = (
        position
        * dataset["next_day_return"]
    )

    turnover = (
        position.diff()
        .abs()
        .fillna(position.abs())
    )

    costs = (
        turnover
        * transaction_cost
    )

    strategy_return = (
        daily_return
        - costs
    )

    equity = (
        INITIAL_CAPITAL
        * (1 + strategy_return).cumprod()
    )

    final_value = equity.iloc[-1]

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1
    )

    years = (
        (
            dataset["Date"].iloc[-1]
            - dataset["Date"].iloc[0]
        ).days
        / 365.25
    )

    annualized_return = (
        (final_value / INITIAL_CAPITAL)
        ** (1 / years)
        - 1
    )

    return {
        "holding_days": holding_days,
        "transaction_cost": transaction_cost,
        "final_value": final_value,
        "return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_ratio(strategy_return),
        "trades": int(turnover.sum()),
        "exposure": position.mean()
    }


def main():

    print("Loading V14 features...")

    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"])

    spy = pd.read_csv(SPY_FILE)
    spy["Date"] = pd.to_datetime(spy["Date"])

    spy = spy.sort_values("Date")

    spy["next_day_return"] = (
        spy["Close"].shift(-1)
        / spy["Close"]
        - 1
    )

    data = (
        data
        .sort_values("Date")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    data = data.merge(
        spy[
            [
                "Date",
                "next_day_return"
            ]
        ],
        on="Date",
        how="inner"
    )

    data = (
        data
        .dropna(
            subset=["next_day_return"]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    features = [
        c for c in data.columns
        if c not in [
            "Date",
            "future_5d_return",
            "next_day_return"
        ]
    ]

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Generating strictly "
        f"out-of-sample predictions..."
    )

    results = generate_predictions(
        data,
        features
    )

    results = results.sort_values(
        "Date"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # FINAL TEST ONLY
    #
    # 2024-2026
    #
    # This remains untouched by parameter selection.
    # --------------------------------------------------

    final_test = results[
        results["Date"].dt.year >= 2024
    ].copy()

    print()
    print("================================")
    print("          V18 ROBUSTNESS")
    print("================================")

    print(
        f"Period: "
        f"{final_test['Date'].min().date()} → "
        f"{final_test['Date'].max().date()}"
    )

    print(
        f"Threshold: "
        f"{THRESHOLD:.2%}"
    )

    print()

    output = []

    for holding_days in HOLDING_PERIODS:

        for transaction_cost in TRANSACTION_COSTS:

            result = run_strategy(
                final_test,
                holding_days,
                transaction_cost
            )

            output.append(result)

    summary = pd.DataFrame(output)

    print(
        summary.to_string(
            index=False,
            formatters={
                "holding_days":
                    "{:.0f}".format,

                "transaction_cost":
                    "{:.2%}".format,

                "final_value":
                    "${:,.2f}".format,

                "return":
                    "{:.2%}".format,

                "annualized_return":
                    "{:.2%}".format,

                "max_drawdown":
                    "{:.2%}".format,

                "sharpe":
                    "{:.3f}".format,

                "exposure":
                    "{:.1%}".format
            }
        )
    )

    # --------------------------------------------------
    # BUY & HOLD
    # --------------------------------------------------

    bh_returns = (
        final_test["next_day_return"]
    )

    bh_equity = (
        INITIAL_CAPITAL
        * (1 + bh_returns).cumprod()
    )

    bh_final = bh_equity.iloc[-1]

    bh_return = (
        bh_final
        / INITIAL_CAPITAL
        - 1
    )

    bh_years = (
        (
            final_test["Date"].iloc[-1]
            - final_test["Date"].iloc[0]
        ).days
        / 365.25
    )

    bh_annualized = (
        (bh_final / INITIAL_CAPITAL)
        ** (1 / bh_years)
        - 1
    )

    bh_sharpe = sharpe_ratio(
        bh_returns
    )

    print()
    print("================================")
    print("   FINAL TEST BUY & HOLD")
    print("================================")

    print(
        f"Final value: "
        f"${bh_final:,.2f}"
    )

    print(
        f"Return: "
        f"{bh_return:.2%}"
    )

    print(
        f"Annualized return: "
        f"{bh_annualized:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{max_drawdown(bh_equity):.2%}"
    )

    print(
        f"Sharpe: "
        f"{bh_sharpe:.3f}"
    )


if __name__ == "__main__":
    main()