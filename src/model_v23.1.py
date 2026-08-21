import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)


# ============================================================
# V23.1 CONFIG
# ============================================================

FEATURE_FILE = Path(
    "data/market_features_v14.csv"
)

SPY_FILE = Path(
    "data/SPY.csv"
)

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005
# 0.05% per side

PROFIT_TARGET = 0.0020
# +0.20% 5-day return defines a "good" trade

CONFIDENCE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

HOLDING_PERIODS = [
    1,
    3,
    5,
    10,
]

MIN_DEVELOPMENT_TRADES = 20

RAW_PRICE_FEATURES = {
    "QQQ_close",
    "IWM_close",
    "DIA_close",
    "VIX_level",
}


# ============================================================
# HELPERS
# ============================================================

def max_drawdown(equity):

    equity = pd.Series(equity)

    peak = equity.cummax()

    drawdown = (
        equity / peak - 1.0
    )

    return drawdown.min()


def sharpe_ratio(returns):

    returns = pd.Series(
        returns,
        dtype=float
    )

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return (
        np.sqrt(252)
        * returns.mean()
        / std
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_features(features):

    dangerous_words = [
        "future",
        "target",
        "label",
        "reward",
    ]

    leaks = []

    for feature in features:

        lower = feature.lower()

        for word in dangerous_words:

            if word in lower:
                leaks.append(feature)
                break

    if leaks:

        print()
        print("================================")
        print("       V23.1 LEAKAGE ERROR")
        print("================================")

        for feature in leaks:
            print(
                f"!!! {feature}"
            )

        raise RuntimeError(
            "Potential future/target leakage detected."
        )

    print()
    print(
        "Leakage check: PASSED"
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=15,
        subsample=0.8,
        random_state=42,
    )


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

def calculate_sample_weights(
    returns
):

    returns = np.asarray(
        returns
    )

    magnitude = np.abs(
        returns
    )

    normalized = np.clip(
        magnitude / 0.02,
        0,
        2,
    )

    return (
        1.0
        + 0.50 * normalized
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("================================")
    print("       V23.1 DATA LOAD")
    print("================================")

    data = pd.read_csv(
        FEATURE_FILE
    )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    spy = pd.read_csv(
        SPY_FILE
    )

    spy["Date"] = pd.to_datetime(
        spy["Date"]
    )

    spy = (
        spy
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Classification target
    # --------------------------------------------------------

    data["trade_reward"] = (
        data["future_5d_return"]
    )

    data["trade_label"] = (
        data["future_5d_return"]
        > PROFIT_TARGET
    ).astype(int)

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    EXCLUDE = {
        "Date",
        "target",

        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return",

        "trade_reward",
        "trade_label",
    }

    features = [
        column
        for column in data.columns
        if column not in EXCLUDE
        and column not in RAW_PRICE_FEATURES
    ]

    check_features(
        features
    )

    # --------------------------------------------------------
    # Merge actual SPY prices
    # --------------------------------------------------------

    price_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    data = data.merge(
        spy[price_columns],
        on="Date",
        how="inner",
    )

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=(
                features
                + [
                    "trade_label",
                    "future_5d_return",
                    "Open",
                    "Close",
                ]
            )
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Date range: "
        f"{data['Date'].min().date()} "
        f"→ "
        f"{data['Date'].max().date()}"
    )

    return data, features


# ============================================================
# WALK-FORWARD PREDICTIONS
# ============================================================

def generate_predictions(
    data,
    features,
    start_year,
    end_year,
):

    predictions = []

    for test_year in range(
        start_year,
        end_year + 1,
    ):

        train = data[
            data["Date"].dt.year
            < test_year
        ].copy()

        test = data[
            data["Date"].dt.year
            == test_year
        ].copy()

        if train.empty:
            continue

        if test.empty:
            continue

        print()
        print(
            f"Training through "
            f"{test_year - 1}..."
        )

        print(
            f"Training rows: "
            f"{len(train)}"
        )

        print(
            f"Test rows: "
            f"{len(test)}"
        )

        model = create_model()

        weights = (
            calculate_sample_weights(
                train[
                    "trade_reward"
                ]
            )
        )

        model.fit(
            train[features],
            train["trade_label"],
            sample_weight=weights,
        )

        probabilities = (
            model
            .predict_proba(
                test[features]
            )[:, 1]
        )

        test = test.copy()

        test["probability"] = (
            probabilities
        )

        predictions.append(
            test[
                [
                    "Date",
                    "Open",
                    "Close",
                    "future_5d_return",
                    "trade_label",
                    "probability",
                ]
            ]
        )

    if not predictions:

        raise RuntimeError(
            "No walk-forward predictions."
        )

    return (
        pd.concat(
            predictions,
            ignore_index=True,
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ============================================================
# CREATE TRADE SIGNALS
# ============================================================

def create_signals(
    predictions,
    threshold,
):

    result = predictions.copy()

    result["signal"] = (
        result["probability"]
        >= threshold
    ).astype(int)

    return result


# ============================================================
# REALISTIC TRADE SIMULATION
# ============================================================

def simulate_strategy(
    predictions,
    threshold,
    holding_days,
    transaction_cost,
):

    data = create_signals(
        predictions,
        threshold,
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    capital = INITIAL_CAPITAL

    equity_curve = []

    daily_returns = []

    trade_records = []

    i = 0

    while i < len(data):

        # ----------------------------------------------------
        # No signal → remain in cash.
        # ----------------------------------------------------

        if data.iloc[i]["signal"] != 1:

            equity_curve.append(
                capital
            )

            daily_returns.append(
                0.0
            )

            i += 1

            continue

        # ----------------------------------------------------
        # Signal occurs on day T.
        #
        # Enter at next trading day's OPEN.
        # ----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(data):

            break

        exit_index = min(
            entry_index
            + holding_days
            - 1,
            len(data) - 1,
        )

        entry_open = float(
            data.iloc[
                entry_index
            ]["Open"]
        )

        exit_close = float(
            data.iloc[
                exit_index
            ]["Close"]
        )

        if entry_open <= 0:

            i += 1
            continue

        # ----------------------------------------------------
        # Gross trade return.
        # ----------------------------------------------------

        gross_return = (
            exit_close
            / entry_open
            - 1.0
        )

        # ----------------------------------------------------
        # Entry + exit transaction costs.
        # ----------------------------------------------------

        net_growth = (
            (1.0 - transaction_cost)
            * (1.0 + gross_return)
            * (1.0 - transaction_cost)
        )

        trade_return = (
            net_growth - 1.0
        )

        starting_capital = capital

        capital *= net_growth

        trade_records.append({
            "signal_date":
                data.iloc[i]["Date"],

            "entry_date":
                data.iloc[entry_index]["Date"],

            "exit_date":
                data.iloc[exit_index]["Date"],

            "probability":
                data.iloc[i]["probability"],

            "gross_return":
                gross_return,

            "net_return":
                trade_return,

            "starting_capital":
                starting_capital,

            "ending_capital":
                capital,
        })

        # ----------------------------------------------------
        # Mark the portfolio at the exit.
        #
        # We record the trade return as one observation for
        # performance analysis.
        # ----------------------------------------------------

        equity_curve.append(
            capital
        )

        daily_returns.append(
            trade_return
        )

        # ----------------------------------------------------
        # No overlapping positions.
        # ----------------------------------------------------

        i = exit_index + 1

    if not trade_records:

        return {
            "threshold":
                threshold,

            "holding_days":
                holding_days,

            "final_value":
                INITIAL_CAPITAL,

            "return":
                0.0,

            "max_drawdown":
                0.0,

            "sharpe":
                0.0,

            "trades":
                0,

            "win_rate":
                0.0,

            "average_trade":
                0.0,

            "profit_factor":
                0.0,

            "equity":
                pd.Series(
                    [INITIAL_CAPITAL]
                ),

            "trades_data":
                pd.DataFrame(),
        }

    trade_df = pd.DataFrame(
        trade_records
    )

    equity = pd.Series(
        equity_curve
    )

    final_value = (
        equity.iloc[-1]
    )

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1.0
    )

    years = (
        (
            data["Date"].iloc[-1]
            - data["Date"].iloc[0]
        ).days
        / 365.25
    )

    if years > 0:

        annualized_return = (
            (
                final_value
                / INITIAL_CAPITAL
            )
            ** (1.0 / years)
            - 1.0
        )

    else:

        annualized_return = 0.0

    trade_returns = pd.Series(
        daily_returns
    )

    wins = (
        trade_returns > 0
    ).sum()

    losses = (
        trade_returns < 0
    ).sum()

    win_rate = (
        wins
        / len(trade_returns)
    )

    average_trade = (
        trade_returns.mean()
    )

    gross_profit = (
        trade_returns[
            trade_returns > 0
        ].sum()
    )

    gross_loss = abs(
        trade_returns[
            trade_returns < 0
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    return {
        "threshold":
            threshold,

        "holding_days":
            holding_days,

        "final_value":
            final_value,

        "return":
            total_return,

        "annualized_return":
            annualized_return,

        "max_drawdown":
            max_drawdown(equity),

        "sharpe":
            sharpe_ratio(
                trade_returns
            ),

        "trades":
            len(trade_df),

        "win_rate":
            win_rate,

        "average_trade":
            average_trade,

        "profit_factor":
            profit_factor,

        "equity":
            equity,

        "trades_data":
            trade_df,
    }


# ============================================================
# DEVELOPMENT THRESHOLD SELECTION
# ============================================================

def select_threshold(
    development_predictions,
):

    print()
    print("================================")
    print("      DEVELOPMENT SEARCH")
    print("================================")

    rows = []

    for threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        # Use 5-day holding during development.
        result = simulate_strategy(
            development_predictions,
            threshold,
            holding_days=5,
            transaction_cost=TRANSACTION_COST,
        )

        rows.append(
            result
        )

        print(
            f"Threshold "
            f"{threshold:.0%}: "
            f"Return "
            f"{result['return']:.2%}, "
            f"Sharpe "
            f"{result['sharpe']:.3f}, "
            f"Trades "
            f"{result['trades']}"
        )

    eligible = [
        result
        for result in rows
        if result["trades"]
        >= MIN_DEVELOPMENT_TRADES
    ]

    if not eligible:

        raise RuntimeError(
            "No development threshold "
            "produced enough trades."
        )

    # Select based on Sharpe.
    best = max(
        eligible,
        key=lambda x: x["sharpe"],
    )

    print()
    print(
        f"Selected threshold: "
        f"{best['threshold']:.0%}"
    )

    print(
        f"Development return: "
        f"{best['return']:.2%}"
    )

    print(
        f"Development Sharpe: "
        f"{best['sharpe']:.3f}"
    )

    return best["threshold"]


# ============================================================
# FINAL TEST
# ============================================================

def run_final_test(
    final_predictions,
    threshold,
):

    print()
    print("================================")
    print("        V23.1 FINAL TEST")
    print("================================")

    for holding_days in HOLDING_PERIODS:

        result = simulate_strategy(
            final_predictions,
            threshold,
            holding_days,
            TRANSACTION_COST,
        )

        profit_factor = (
            result["profit_factor"]
        )

        if np.isinf(
            profit_factor
        ):

            pf_text = "inf"

        else:

            pf_text = (
                f"{profit_factor:.3f}"
            )

        print(
            f"\nHolding: "
            f"{holding_days} days"
        )

        print(
            f"Final value: "
            f"${result['final_value']:,.2f}"
        )

        print(
            f"Return: "
            f"{result['return']:.2%}"
        )

        print(
            f"Annualized: "
            f"{result['annualized_return']:.2%}"
        )

        print(
            f"Max drawdown: "
            f"{result['max_drawdown']:.2%}"
        )

        print(
            f"Sharpe: "
            f"{result['sharpe']:.3f}"
        )

        print(
            f"Trades: "
            f"{result['trades']}"
        )

        print(
            f"Win rate: "
            f"{result['win_rate']:.2%}"
        )

        print(
            f"Average trade: "
            f"{result['average_trade']:.2%}"
        )

        print(
            f"Profit factor: "
            f"{pf_text}"
        )


# ============================================================
# BUY & HOLD BENCHMARK
# ============================================================

def buy_and_hold(
    predictions
):

    data = (
        predictions
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Actual SPY daily close-to-close returns.
    closes = (
        data["Close"]
        .astype(float)
    )

    daily_returns = (
        closes.pct_change()
        .fillna(0.0)
    )

    equity = (
        INITIAL_CAPITAL
        * (1.0 + daily_returns)
        .cumprod()
    )

    final_value = (
        equity.iloc[-1]
    )

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1.0
    )

    years = (
        (
            data["Date"].iloc[-1]
            - data["Date"].iloc[0]
        ).days
        / 365.25
    )

    annualized = (
        (
            final_value
            / INITIAL_CAPITAL
        )
        ** (1.0 / years)
        - 1.0
    )

    return {
        "final_value":
            final_value,

        "return":
            total_return,

        "annualized":
            annualized,

        "max_drawdown":
            max_drawdown(equity),

        "sharpe":
            sharpe_ratio(
                daily_returns
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "############################################################"
    )

    print(
        "#                       V23.1"
    )

    print(
        "#             REAL PRICE BACKTEST"
    )

    print(
        "############################################################"
    )

    data, features = load_data()

    # --------------------------------------------------------
    # Generate OOS development predictions.
    # --------------------------------------------------------

    development_predictions = (
        generate_predictions(
            data,
            features,
            start_year=2020,
            end_year=2023,
        )
    )

    # --------------------------------------------------------
    # Select threshold ONLY using 2020-2023.
    # --------------------------------------------------------

    threshold = select_threshold(
        development_predictions
    )

    # --------------------------------------------------------
    # Generate FINAL TEST predictions separately.
    #
    # For each year, the model is trained only on years
    # before that year.
    # --------------------------------------------------------

    final_predictions = (
        generate_predictions(
            data,
            features,
            start_year=2024,
            end_year=2026,
        )
    )

    print()
    print(
        "Final-test prediction count: "
        f"{len(final_predictions)}"
    )

    # Save predictions for later inspection.
    prediction_file = Path(
        "data/v23_1_oos_predictions.csv"
    )

    final_predictions.to_csv(
        prediction_file,
        index=False,
    )

    print(
        f"Saved final-test predictions to "
        f"{prediction_file}"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    run_final_test(
        final_predictions,
        threshold,
    )

    # --------------------------------------------------------
    # BUY & HOLD
    # --------------------------------------------------------

    benchmark = buy_and_hold(
        final_predictions
    )

    print()
    print(
        "================================"
    )

    print(
        "       BUY & HOLD"
    )

    print(
        "================================"
    )

    print(
        f"Final value: "
        f"${benchmark['final_value']:,.2f}"
    )

    print(
        f"Return: "
        f"{benchmark['return']:.2%}"
    )

    print(
        f"Annualized: "
        f"{benchmark['annualized']:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{benchmark['max_drawdown']:.2%}"
    )

    print(
        f"Sharpe: "
        f"{benchmark['sharpe']:.3f}"
    )

    print()
    print(
        "############################################################"
    )

    print(
        "#                    V23.1 COMPLETE"
    )

    print(
        "############################################################"
    )


if __name__ == "__main__":
    main()