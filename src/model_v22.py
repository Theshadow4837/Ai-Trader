import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)


# ============================================================
# V22 CONFIG
# ============================================================

DATA_FILE = Path("data/market_features_v14.csv")

INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.0005

# A trade is considered a "good" trade if the 5-day return
# exceeds this threshold.
PROFIT_TARGET = 0.0020       # +0.20%

# Probability thresholds tested ONLY during development.
CONFIDENCE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

# Minimum number of trades required when selecting a threshold.
MIN_DEVELOPMENT_TRADES = 30


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def max_drawdown(equity):
    equity = pd.Series(equity)

    peak = equity.cummax()

    drawdown = (
        equity / peak - 1.0
    )

    return drawdown.min()


def sharpe_ratio(returns):
    returns = pd.Series(returns)

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

def check_for_leakage(columns, feature_columns):

    dangerous_exact = {
        "target",
        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return",
        "trade_reward",
        "trade_label",
    }

    dangerous_words = [
        "future",
        "target",
        "label",
        "reward",
    ]

    leaks = []

    for column in feature_columns:

        lower = column.lower()

        if lower in dangerous_exact:
            leaks.append(column)
            continue

        # Catch accidental future-derived columns.
        for word in dangerous_words:

            if word in lower:
                leaks.append(column)
                break

    if leaks:

        print_header("V22 LEAKAGE DETECTED")

        print(
            "The following columns were about to be used "
            "as model features:"
        )

        for column in leaks:
            print(f"  !!! {column}")

        raise RuntimeError(
            "V22 stopped because potential target leakage "
            "was detected."
        )

    print()
    print("Leakage check: PASSED")
    print(f"Features checked: {len(feature_columns)}")


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
        random_state=42
    )


# ============================================================
# REWARD WEIGHTS
# ============================================================

def calculate_sample_weights(returns):

    returns = np.asarray(returns)

    # Reward stronger correct outcomes slightly more.
    #
    # Example:
    # +0.5% return gets more weight than +0.05%.
    #
    # Clipping prevents extreme market days from dominating
    # the entire model.

    magnitude = np.abs(returns)

    normalized = np.clip(
        magnitude / 0.02,
        0,
        2
    )

    weights = (
        1.0
        + 0.50 * normalized
    )

    return weights


# ============================================================
# PREPARE DATA
# ============================================================

def load_data():

    print_header("LOADING V22 DATA")

    data = pd.read_csv(DATA_FILE)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .reset_index(drop=True)
    )

    required_columns = [
        "Date",
        "future_5d_return",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # Create the classification target.
    #
    # 1 = future 5-day return exceeded +0.20%
    # 0 = otherwise

    data["trade_reward"] = (
        data["future_5d_return"]
    )

    data["trade_label"] = (
        data["trade_reward"]
        > PROFIT_TARGET
    ).astype(int)

    # --------------------------------------------------------
    # Features
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

    FEATURES = [
        column
        for column in data.columns
        if column not in EXCLUDE
    ]

    check_for_leakage(
        data.columns,
        FEATURES
    )

    # Remove rows that contain missing feature values.
    data = (
        data
        .dropna(
            subset=FEATURES
            + [
                "future_5d_return",
                "trade_label"
            ]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print()
    print(f"Rows: {len(data)}")
    print(f"Features: {len(FEATURES)}")

    print(
        f"Date range: "
        f"{data['Date'].min().date()} → "
        f"{data['Date'].max().date()}"
    )

    print()
    print("Target distribution:")

    print(
        data["trade_label"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .rename({
            0: "No trade-quality outcome",
            1: "Good trade"
        })
        .to_string()
    )

    return data, FEATURES


# ============================================================
# WALK-FORWARD DEVELOPMENT
# ============================================================

def generate_walk_forward_predictions(
    data,
    features,
    start_year,
    end_year
):

    predictions = []

    print_header(
        f"V22 WALK-FORWARD: "
        f"{start_year}-{end_year}"
    )

    for test_year in range(
        start_year,
        end_year + 1
    ):

        train = data[
            data["Date"].dt.year < test_year
        ].copy()

        test = data[
            data["Date"].dt.year == test_year
        ].copy()

        if len(train) == 0:
            continue

        if len(test) == 0:
            continue

        print(
            f"\nTraining through "
            f"{test_year - 1}..."
        )

        print(
            f"Training samples: "
            f"{len(train)}"
        )

        print(
            f"Test samples: "
            f"{len(test)}"
        )

        model = create_model()

        sample_weights = (
            calculate_sample_weights(
                train["trade_reward"]
            )
        )

        model.fit(
            train[features],
            train["trade_label"],
            sample_weight=sample_weights
        )

        probabilities = model.predict_proba(
            test[features]
        )[:, 1]

        predictions.append(
            pd.DataFrame({
                "Date":
                    test["Date"].values,

                "actual_label":
                    test["trade_label"].values,

                "actual_return":
                    test["future_5d_return"].values,

                "probability":
                    probabilities,

                "year":
                    test_year
            })
        )

    if not predictions:

        raise RuntimeError(
            "No walk-forward predictions generated."
        )

    return pd.concat(
        predictions,
        ignore_index=True
    )


# ============================================================
# EVALUATE CLASSIFICATION
# ============================================================

def evaluate_predictions(
    predictions,
    probability_threshold
):

    df = predictions.copy()

    df["predicted_label"] = (
        df["probability"]
        >= probability_threshold
    ).astype(int)

    accuracy = accuracy_score(
        df["actual_label"],
        df["predicted_label"]
    )

    balanced = balanced_accuracy_score(
        df["actual_label"],
        df["predicted_label"]
    )

    trades = (
        df["predicted_label"] == 1
    )

    strong_samples = int(
        trades.sum()
    )

    if strong_samples > 0:

        average_reward = (
            df.loc[
                trades,
                "actual_return"
            ].mean()
        )

        strong_accuracy = (
            df.loc[
                trades,
                "actual_label"
            ].mean()
        )

    else:

        average_reward = 0.0
        strong_accuracy = 0.0

    return {
        "threshold":
            probability_threshold,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced,

        "average_reward":
            average_reward,

        "strong_accuracy":
            strong_accuracy,

        "strong_samples":
            strong_samples,

        "samples":
            len(df)
    }


# ============================================================
# DEVELOPMENT THRESHOLD SEARCH
# ============================================================

def select_threshold(
    development_predictions
):

    print_header(
        "V22 DEVELOPMENT THRESHOLD SEARCH"
    )

    rows = []

    for threshold in CONFIDENCE_THRESHOLDS:

        result = evaluate_predictions(
            development_predictions,
            threshold
        )

        rows.append(result)

    summary = pd.DataFrame(rows)

    print()

    print(
        summary.to_string(
            index=False,
            formatters={
                "threshold":
                    "{:.0%}".format,

                "accuracy":
                    "{:.4f}".format,

                "balanced_accuracy":
                    "{:.4f}".format,

                "average_reward":
                    "{:.4%}".format,

                "strong_accuracy":
                    "{:.4f}".format,
            }
        )
    )

    eligible = summary[
        summary["strong_samples"]
        >= MIN_DEVELOPMENT_TRADES
    ].copy()

    if eligible.empty:

        raise RuntimeError(
            "No confidence threshold produced "
            "enough development trades."
        )

    # Select using average reward first.
    #
    # This means we're optimizing the thing we actually
    # care about rather than simply maximizing accuracy.

    best = eligible.loc[
        eligible["average_reward"].idxmax()
    ]

    selected_threshold = (
        float(best["threshold"])
    )

    print()

    print(
        f"Selected confidence threshold: "
        f"{selected_threshold:.0%}"
    )

    print(
        f"Development average reward: "
        f"{best['average_reward']:.4%}"
    )

    print(
        f"Development strong-signal accuracy: "
        f"{best['strong_accuracy']:.4f}"
    )

    return selected_threshold, summary


# ============================================================
# FINAL PAPER-TEST STYLE EVALUATION
# ============================================================

def run_final_test(
    data,
    features,
    threshold
):

    print_header(
        "V22 FINAL TEST: 2024-2026"
    )

    # IMPORTANT:
    #
    # The model is trained ONLY on data through 2023.
    #
    # Therefore 2024-2026 remains an untouched block.

    train = data[
        data["Date"].dt.year <= 2023
    ].copy()

    final_test = data[
        data["Date"].dt.year >= 2024
    ].copy()

    print(
        f"Training period: "
        f"{train['Date'].min().date()} → "
        f"{train['Date'].max().date()}"
    )

    print(
        f"Final test period: "
        f"{final_test['Date'].min().date()} → "
        f"{final_test['Date'].max().date()}"
    )

    print(
        f"Training samples: "
        f"{len(train)}"
    )

    print(
        f"Final test samples: "
        f"{len(final_test)}"
    )

    model = create_model()

    sample_weights = (
        calculate_sample_weights(
            train["trade_reward"]
        )
    )

    model.fit(
        train[features],
        train["trade_label"],
        sample_weight=sample_weights
    )

    final_test = final_test.copy()

    final_test["probability"] = (
        model.predict_proba(
            final_test[features]
        )[:, 1]
    )

    final_test["signal"] = (
        final_test["probability"]
        >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        final_test["trade_label"],
        final_test["signal"]
    )

    balanced = balanced_accuracy_score(
        final_test["trade_label"],
        final_test["signal"]
    )

    strong = (
        final_test["signal"] == 1
    )

    strong_samples = int(
        strong.sum()
    )

    if strong_samples > 0:

        strong_accuracy = (
            final_test.loc[
                strong,
                "trade_label"
            ].mean()
        )

        average_reward = (
            final_test.loc[
                strong,
                "future_5d_return"
            ].mean()
        )

    else:

        strong_accuracy = 0.0
        average_reward = 0.0

    # --------------------------------------------------------
    # Strategy simulation
    #
    # This is NOT live trading.
    #
    # Each signal gets the actual 5-day return associated
    # with that prediction.
    #
    # We subtract one transaction cost per signal.
    # --------------------------------------------------------

    final_test["strategy_return"] = np.where(
        final_test["signal"] == 1,
        final_test["future_5d_return"]
        - TRANSACTION_COST,
        0.0
    )

    equity = (
        INITIAL_CAPITAL
        * (
            1
            + final_test["strategy_return"]
        ).cumprod()
    )

    final_value = equity.iloc[-1]

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1
    )

    years = (
        (
            final_test["Date"].iloc[-1]
            - final_test["Date"].iloc[0]
        ).days
        / 365.25
    )

    annualized_return = (
        (
            final_value
            / INITIAL_CAPITAL
        )
        ** (1 / years)
        - 1
    )

    mdd = max_drawdown(
        equity
    )

    sharpe = sharpe_ratio(
        final_test["strategy_return"]
    )

    trades = int(
        final_test["signal"].sum()
    )

    exposure = (
        final_test["signal"].mean()
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced:.4f}"
    )

    print(
        f"Strong-signal accuracy: "
        f"{strong_accuracy:.4f}"
    )

    print(
        f"Average reward per trade: "
        f"{average_reward:.4%}"
    )

    print(
        f"Strong signals: "
        f"{strong_samples}"
    )

    print()

    print(
        "================================"
    )

    print(
        "       V22 FINAL STRATEGY"
    )

    print(
        "================================"
    )

    print(
        f"Initial capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final value: "
        f"${final_value:,.2f}"
    )

    print(
        f"Return: "
        f"{total_return:.2%}"
    )

    print(
        f"Annualized return: "
        f"{annualized_return:.2%}"
    )

    print(
        f"Max drawdown: "
        f"{mdd:.2%}"
    )

    print(
        f"Sharpe: "
        f"{sharpe:.3f}"
    )

    print(
        f"Trades: "
        f"{trades}"
    )

    print(
        f"Exposure: "
        f"{exposure:.1%}"
    )

    # --------------------------------------------------------
    # Year-by-year final test
    # --------------------------------------------------------

    print()

    print(
        "================================"
    )

    print(
        "   FINAL TEST BY YEAR"
    )

    print(
        "================================"
    )

    yearly = []

    for year, group in final_test.groupby(
        final_test["Date"].dt.year
    ):

        signals = (
            group["signal"] == 1
        )

        signal_count = int(
            signals.sum()
        )

        if signal_count > 0:

            reward = (
                group.loc[
                    signals,
                    "future_5d_return"
                ].mean()
            )

            signal_accuracy = (
                group.loc[
                    signals,
                    "trade_label"
                ].mean()
            )

        else:

            reward = 0.0
            signal_accuracy = 0.0

        yearly.append({
            "year":
                year,

            "accuracy":
                accuracy_score(
                    group["trade_label"],
                    group["signal"]
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    group["trade_label"],
                    group["signal"]
                ),

            "average_reward":
                reward,

            "strong_accuracy":
                signal_accuracy,

            "strong_samples":
                signal_count,

            "samples":
                len(group)
        })

    yearly_summary = pd.DataFrame(
        yearly
    )

    print()

    print(
        yearly_summary.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.4f}".format,

                "balanced_accuracy":
                    "{:.4f}".format,

                "average_reward":
                    "{:.4%}".format,

                "strong_accuracy":
                    "{:.4f}".format
            }
        )
    )

    # --------------------------------------------------------
    # Buy & hold comparison
    # --------------------------------------------------------
    #
    # This uses the same 2024-2026 dates but compounds the
    # actual 5-day target returns. It is a rough benchmark,
    # NOT a day-by-day SPY buy-and-hold reconstruction.
    #
    # The strategy should ultimately be compared against
    # the proper daily SPY benchmark in the backtester.
    # --------------------------------------------------------

    bh_returns = (
        final_test["future_5d_return"]
    )

    bh_equity = (
        INITIAL_CAPITAL
        * (
            1 + bh_returns
        ).cumprod()
    )

    bh_final = bh_equity.iloc[-1]

    bh_return = (
        bh_final
        / INITIAL_CAPITAL
        - 1
    )

    print()

    print(
        "================================"
    )

    print(
        "   V22 BENCHMARK"
    )

    print(
        "================================"
    )

    print(
        f"Compounded target benchmark: "
        f"${bh_final:,.2f}"
    )

    print(
        f"Benchmark return: "
        f"{bh_return:.2%}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        final_test["trade_label"],
        final_test["signal"]
    )

    print()

    print(
        "================================"
    )

    print(
        "      CONFUSION MATRIX"
    )

    print(
        "================================"
    )

    print(
        "Rows = actual"
    )

    print(
        "Columns = predicted"
    )

    print(cm)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature":
            features,

        "importance":
            model.feature_importances_
    })

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .head(15)
    )

    print()

    print(
        "================================"
    )

    print(
        "     TOP 15 FEATURES"
    )

    print(
        "================================"
    )

    print(
        importance.to_string(
            index=False,
            formatters={
                "importance":
                    "{:.6f}".format
            }
        )
    )

    return final_test


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("############################################################")
    print("#                         V22")
    print("#              TRADE QUALITY CLASSIFIER")
    print("############################################################")

    data, features = load_data()

    # --------------------------------------------------------
    # DEVELOPMENT WALK-FORWARD
    #
    # 2020-2023 is allowed to influence threshold selection.
    # --------------------------------------------------------

    development_predictions = (
        generate_walk_forward_predictions(
            data,
            features,
            start_year=2020,
            end_year=2023
        )
    )

    print_header(
        "V22 DEVELOPMENT RESULTS"
    )

    development_rows = []

    for threshold in CONFIDENCE_THRESHOLDS:

        development_rows.append(
            evaluate_predictions(
                development_predictions,
                threshold
            )
        )

    development_summary = pd.DataFrame(
        development_rows
    )

    print()

    print(
        development_summary.to_string(
            index=False,
            formatters={
                "threshold":
                    "{:.0%}".format,

                "accuracy":
                    "{:.4f}".format,

                "balanced_accuracy":
                    "{:.4f}".format,

                "average_reward":
                    "{:.4%}".format,

                "strong_accuracy":
                    "{:.4f}".format
            }
        )
    )

    # --------------------------------------------------------
    # SELECT THRESHOLD
    # --------------------------------------------------------

    best_threshold, _ = (
        select_threshold(
            development_predictions
        )
    )

    # --------------------------------------------------------
    # FINAL TEST
    #
    # 2024-2026 has NOT been used for threshold selection.
    # --------------------------------------------------------

    run_final_test(
        data,
        features,
        best_threshold
    )

    print()
    print("############################################################")
    print("#                         V22 DONE")
    print("############################################################")

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "2024-2026 was kept out of threshold selection."
    )

    print(
        "If V22 performs poorly on the final test, "
        "we do NOT tune it against those results."
    )

    print()


if __name__ == "__main__":
    main()