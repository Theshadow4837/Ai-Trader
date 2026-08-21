import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor


DATA_FILE = Path("data/market_features_v14.csv")
SPY_FILE = Path("data/SPY.csv")

INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.0005

THRESHOLDS = [
    0.0000,
    0.0010,
    0.0020,
    0.0030,
    0.0050,
]


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
        spy[["Date", "next_day_return"]],
        on="Date",
        how="inner"
    )

    data = (
        data
        .dropna(subset=["next_day_return"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    FEATURES = [
        c for c in data.columns
        if c not in [
            "Date",
            "future_5d_return",
            "next_day_return"
        ]
    ]

    print(f"Features: {len(FEATURES)}")
    print(
        f"Testing period: "
        f"{data['Date'].min().date()} → "
        f"{data['Date'].max().date()}"
    )

    # --------------------------------------------------
    # Generate strictly out-of-sample predictions
    # --------------------------------------------------

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
            train[FEATURES],
            train["future_5d_return"]
        )

        test["prediction"] = model.predict(
            test[FEATURES]
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

    results = pd.concat(
        predictions,
        ignore_index=True
    )

    results = results.sort_values(
        "Date"
    ).reset_index(drop=True)

        # --------------------------------------------------
    # V16 DEVELOPMENT / FINAL TEST SPLIT
    #
    # 2020-2023 = threshold selection
    # 2024-2026 = untouched final test
    # --------------------------------------------------

    development = results[
        results["Date"].dt.year <= 2023
    ].copy()

    final_test = results[
        results["Date"].dt.year >= 2024
    ].copy()

    print()
    print("================================")
    print("          V16 SPLIT")
    print("================================")

    print(
        f"Development: "
        f"{development['Date'].min().date()} → "
        f"{development['Date'].max().date()}"
    )

    print(
        f"Final test: "
        f"{final_test['Date'].min().date()} → "
        f"{final_test['Date'].max().date()}"
    )
    # --------------------------------------------------
    # Prevent same-day look-ahead
        # --------------------------------------------------
    # V16: THRESHOLD SELECTION
    #
    # Development period:
    # 2020-2023
    #
    # Final untouched test:
    # 2024-2026
    # --------------------------------------------------

    HOLD_DAYS = 5

    THRESHOLDS = [
        0.0000,
        0.0010,
        0.0020,
        0.0030,
        0.0050,
    ]

    def run_strategy(dataset, threshold):

        dataset = dataset.reset_index(drop=True)

        position = np.zeros(len(dataset))

        i = 0

        while i < len(dataset):

            prediction = dataset.loc[
                i,
                "prediction"
            ]

            if prediction > threshold:

                start = i + 1

                end = min(
                    start + HOLD_DAYS,
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
            * TRANSACTION_COST
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
            "threshold": threshold,
            "final_value": final_value,
            "return": total_return,
            "annualized_return":
                annualized_return,
            "max_drawdown":
                max_drawdown(equity),
            "sharpe":
                sharpe_ratio(strategy_return),
            "trades":
                int(turnover.sum()),
            "exposure":
                position.mean()
        }


    # --------------------------------------------------
    # STEP 1
    # Choose threshold ONLY using 2020-2023
    # --------------------------------------------------

    print()
    print("================================")
    print("     V16 THRESHOLD SELECTION")
    print("================================")

    development_results = []

    for threshold in THRESHOLDS:

        result = run_strategy(
            development,
            threshold
        )

        development_results.append(
            result
        )

    development_summary = pd.DataFrame(
        development_results
    )

    print()
    print("DEVELOPMENT: 2020-2023")

    print(
        development_summary.to_string(
            index=False,
            formatters={
                "threshold":
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
    # Choose the threshold using DEVELOPMENT ONLY
    #
    # Sharpe is used because we care about risk-adjusted
    # performance, not simply the biggest raw return.
    # --------------------------------------------------

    best_row = development_summary.loc[
        development_summary["sharpe"].idxmax()
    ]

    BEST_THRESHOLD = best_row[
        "threshold"
    ]

    print()
    print(
        f"Selected threshold: "
        f"{BEST_THRESHOLD:.2%}"
    )

    print(
        "This threshold was selected "
        "WITHOUT using 2024-2026."
    )


    # --------------------------------------------------
    # STEP 2
    # Test ONLY the selected threshold on 2024-2026
    # --------------------------------------------------

    final_result = run_strategy(
        final_test,
        BEST_THRESHOLD
    )

    print()
    print("================================")
    print("       V16 FINAL TEST")
    print("================================")

    print(
        "Period: 2024-01-02 → 2026-08-17"
    )

    print(
        f"Threshold: "
        f"{BEST_THRESHOLD:.2%}"
    )

    print(
        f"Final value: "
        f"${final_result['final_value']:,.2f}"
    )

    print(
        f"Return: "
        f"{final_result['return']:.2%}"
    )

    print(
        f"Annualized return: "
        f"{final_result['annualized_return']:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{final_result['max_drawdown']:.2%}"
    )

    print(
        f"Sharpe: "
        f"{final_result['sharpe']:.3f}"
    )

    print(
        f"Trades: "
        f"{final_result['trades']}"
    )

    print(
        f"Exposure: "
        f"{final_result['exposure']:.1%}"
    )


    # --------------------------------------------------
    # FINAL TEST BUY & HOLD
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
        f"{sharpe_ratio(bh_returns):.3f}"
    )
if __name__ == "__main__":
    main()