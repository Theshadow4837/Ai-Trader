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
    # Prevent same-day look-ahead
    #
    # Signal generated on day T is applied beginning
    # on day T+1.
    # --------------------------------------------------

    results["position_signal"] = 0.0

    for threshold in THRESHOLDS:

        results[f"position_{threshold}"] = (
            results["prediction"] > threshold
        ).astype(float)

        results[f"position_{threshold}"] = (
            results[f"position_{threshold}"]
            .shift(1)
            .fillna(0)
        )

    # --------------------------------------------------
    # Backtest each threshold
    # --------------------------------------------------

    print()
    print("================================")
    print("       V14 BACKTEST")
    print("================================")

    print(
        f"Initial capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST:.2%}"
    )

    output = []

    for threshold in THRESHOLDS:

        position_col = (
            f"position_{threshold}"
        )

        position = results[position_col]

        daily_return = (
            position
            * results["next_day_return"]
        )

        turnover = position.diff().abs().fillna(
            position.abs()
        )

        costs = (
            turnover
            * TRANSACTION_COST
        )

        strategy_return = (
            daily_return - costs
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
                results["Date"].iloc[-1]
                - results["Date"].iloc[0]
            ).days
            / 365.25
        )

        annualized_return = (
            (final_value / INITIAL_CAPITAL)
            ** (1 / years)
            - 1
        )

        mdd = max_drawdown(equity)

        sharpe = sharpe_ratio(
            strategy_return
        )

        trades = int(
            turnover.sum()
        )

        exposure = (
            position.mean()
        )

        output.append({
            "threshold": threshold,
            "final_value": final_value,
            "return": total_return,
            "annualized_return":
                annualized_return,
            "max_drawdown": mdd,
            "sharpe": sharpe,
            "trades": trades,
            "exposure": exposure
        })

    summary = pd.DataFrame(output)

    print()

    print(
        summary.to_string(
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
    # Buy & Hold
    # --------------------------------------------------

    buy_hold_returns = (
        results["next_day_return"]
    )

    buy_hold_equity = (
        INITIAL_CAPITAL
        * (1 + buy_hold_returns).cumprod()
    )

    bh_final = buy_hold_equity.iloc[-1]

    bh_return = (
        bh_final
        / INITIAL_CAPITAL
        - 1
    )

    bh_years = (
        (
            results["Date"].iloc[-1]
            - results["Date"].iloc[0]
        ).days
        / 365.25
    )

    bh_annualized = (
        (bh_final / INITIAL_CAPITAL)
        ** (1 / bh_years)
        - 1
    )

    bh_mdd = max_drawdown(
        buy_hold_equity
    )

    bh_sharpe = sharpe_ratio(
        buy_hold_returns
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
        f"{bh_return:.2%}"
    )

    print(
        f"Annualized return: "
        f"{bh_annualized:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{bh_mdd:.2%}"
    )

    print(
        f"Sharpe: "
        f"{bh_sharpe:.3f}"
    )


if __name__ == "__main__":
    main()