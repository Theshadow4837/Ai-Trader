import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)


# ============================================================
# V23 CONFIG
# ============================================================

DATA_FILE = Path("data/market_features_v14.csv")

PROFIT_TARGET = 0.0020       # +0.20%
TRANSACTION_COST = 0.0005
INITIAL_CAPITAL = 10_000.0

CONFIDENCE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]

MIN_DEVELOPMENT_TRADES = 30


# ============================================================
# RAW PRICE LEVELS TO REMOVE
#
# V22 showed DIA_close as its #1 feature.
# We remove raw price levels and keep normalized
# returns / volatility / distance-to-average features.
# ============================================================

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

def check_for_leakage(features):

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
        print("       V23 LEAKAGE ERROR")
        print("================================")

        for feature in leaks:
            print(f"!!! {feature}")

        raise RuntimeError(
            "Potential target leakage detected."
        )

    print()
    print("Leakage check: PASSED")


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

def calculate_sample_weights(returns):

    returns = np.asarray(returns)

    magnitude = np.abs(returns)

    # Mild weighting toward meaningful moves.
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
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("================================")
    print("          V23 DATA")
    print("================================")

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
    )

    # --------------------------------------------------------
    # V23 TARGET
    # --------------------------------------------------------

    data["trade_reward"] = (
        data["future_5d_return"]
    )

    data["trade_label"] = (
        data["future_5d_return"]
        > PROFIT_TARGET
    ).astype(int)

    # --------------------------------------------------------
    # EXCLUDE TARGETS + RAW PRICE LEVELS
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
        c
        for c in data.columns
        if c not in EXCLUDE
        and c not in RAW_PRICE_FEATURES
    ]

    check_for_leakage(
        FEATURES
    )

    data = (
        data
        .dropna(
            subset=FEATURES
            + [
                "future_5d_return",
                "trade_label",
            ]
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Raw price features removed: "
        f"{len(RAW_PRICE_FEATURES)}"
    )

    print(
        f"Date range: "
        f"{data['Date'].min().date()} "
        f"→ "
        f"{data['Date'].max().date()}"
    )

    print()
    print("Removed:")

    for feature in sorted(
        RAW_PRICE_FEATURES
    ):
        print(
            f" - {feature}"
        )

    return data, FEATURES


# ============================================================
# WALK-FORWARD PREDICTIONS
# ============================================================

def walk_forward(
    data,
    features,
    start_year,
    end_year
):

    predictions = []

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

        if train.empty or test.empty:
            continue

        print()
        print(
            f"Training through "
            f"{test_year - 1}..."
        )

        print(
            f"Train: {len(train)}"
        )

        print(
            f"Test: {len(test)}"
        )

        model = create_model()

        weights = calculate_sample_weights(
            train["trade_reward"]
        )

        model.fit(
            train[features],
            train["trade_label"],
            sample_weight=weights
        )

        probabilities = (
            model
            .predict_proba(
                test[features]
            )[:, 1]
        )

        predictions.append(
            pd.DataFrame({
                "Date":
                    test["Date"].values,

                "actual_label":
                    test["trade_label"].values,

                "actual_return":
                    test[
                        "future_5d_return"
                    ].values,

                "probability":
                    probabilities,

                "year":
                    test_year,
            })
        )

    return pd.concat(
        predictions,
        ignore_index=True
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    predictions,
    threshold
):

    df = predictions.copy()

    df["signal"] = (
        df["probability"]
        >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        df["actual_label"],
        df["signal"]
    )

    balanced = balanced_accuracy_score(
        df["actual_label"],
        df["signal"]
    )

    trades = (
        df["signal"] == 1
    )

    trade_count = int(
        trades.sum()
    )

    if trade_count > 0:

        average_reward = (
            df.loc[
                trades,
                "actual_return"
            ].mean()
        )

        trade_accuracy = (
            df.loc[
                trades,
                "actual_label"
            ].mean()
        )

    else:

        average_reward = 0.0
        trade_accuracy = 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "average_reward": average_reward,
        "trade_accuracy": trade_accuracy,
        "trades": trade_count,
        "samples": len(df),
    }


# ============================================================
# DEVELOPMENT THRESHOLD SELECTION
# ============================================================

def select_threshold(
    development
):

    print()
    print("================================")
    print("     V23 DEVELOPMENT")
    print("================================")

    rows = []

    for threshold in (
        CONFIDENCE_THRESHOLDS
    ):

        rows.append(
            evaluate(
                development,
                threshold
            )
        )

    results = pd.DataFrame(
        rows
    )

    print()

    print(
        results.to_string(
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

                "trade_accuracy":
                    "{:.4f}".format,
            }
        )
    )

    eligible = results[
        results["trades"]
        >= MIN_DEVELOPMENT_TRADES
    ]

    if eligible.empty:

        raise RuntimeError(
            "No threshold produced enough trades."
        )

    # Select using trade reward, not raw accuracy.
    best = eligible.loc[
        eligible["average_reward"].idxmax()
    ]

    threshold = float(
        best["threshold"]
    )

    print()
    print(
        f"Selected threshold: "
        f"{threshold:.0%}"
    )

    print(
        f"Development reward: "
        f"{best['average_reward']:.4%}"
    )

    return threshold


# ============================================================
# FINAL TEST
# ============================================================

def final_test(
    data,
    features,
    threshold
):

    print()
    print("================================")
    print("       V23 FINAL TEST")
    print("================================")

    train = data[
        data["Date"].dt.year <= 2023
    ].copy()

    test = data[
        data["Date"].dt.year >= 2024
    ].copy()

    print(
        f"Training: "
        f"{train['Date'].min().date()} "
        f"→ "
        f"{train['Date'].max().date()}"
    )

    print(
        f"Final test: "
        f"{test['Date'].min().date()} "
        f"→ "
        f"{test['Date'].max().date()}"
    )

    model = create_model()

    weights = calculate_sample_weights(
        train["trade_reward"]
    )

    model.fit(
        train[features],
        train["trade_label"],
        sample_weight=weights
    )

    test = test.copy()

    test["probability"] = (
        model.predict_proba(
            test[features]
        )[:, 1]
    )

    test["signal"] = (
        test["probability"]
        >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    accuracy = accuracy_score(
        test["trade_label"],
        test["signal"]
    )

    balanced = balanced_accuracy_score(
        test["trade_label"],
        test["signal"]
    )

    selected = (
        test["signal"] == 1
    )

    trade_count = int(
        selected.sum()
    )

    if trade_count:

        trade_accuracy = (
            test.loc[
                selected,
                "trade_label"
            ].mean()
        )

        average_reward = (
            test.loc[
                selected,
                "future_5d_return"
            ].mean()
        )

    else:

        trade_accuracy = 0.0
        average_reward = 0.0

    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced:.4f}"
    )

    print(
        f"Selected-trade accuracy: "
        f"{trade_accuracy:.4f}"
    )

    print(
        f"Average reward/trade: "
        f"{average_reward:.4%}"
    )

    print(
        f"Trades: {trade_count}"
    )

    # --------------------------------------------------------
    # 5-day strategy simulation
    # --------------------------------------------------------

    # Each selected prediction is treated as one
    # five-trading-day research trade.
    #
    # The next signal is not allowed to stack with an
    # already-active trade.

    position = np.zeros(
        len(test)
    )

    # Reset the index so positional iteration is clean.
    test = test.reset_index(drop=True)

    i = 0

    while i < len(test):

        if test.iloc[i]["signal"] == 1:

            start = i + 1

            end = min(
                start + 5,
                len(test)
            )

            position[start:end] = 1.0

            i = end

        else:
            i += 1

            start = i + 1

            end = min(
                start + 5,
                len(test)
            )

            position[
                start:end
            ] = 1.0

            i = end

    position = pd.Series(
        position,
        index=test.index
    )

    # Use the 5-day target as the return associated with
    # each signal. This remains a model-research metric.
    #
    # We do NOT claim this is an executable daily SPY
    # backtest.

    trade_returns = np.where(
        test["signal"] == 1,
        test["future_5d_return"]
        - TRANSACTION_COST,
        0.0
    )

    equity = (
        INITIAL_CAPITAL
        * (
            1 + trade_returns
        ).cumprod()
    )

    final_value = equity[-1]

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1
    )

    years = (
        (
            test["Date"].iloc[-1]
            - test["Date"].iloc[0]
        ).days
        / 365.25
    )

    annualized = (
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
        trade_returns
    )

    print()
    print(
        "================================"
    )
    print(
        "       V23 STRATEGY"
    )
    print(
        "================================"
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
        f"{annualized:.2%}"
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
        f"Trade signals: "
        f"{trade_count}"
    )

    # --------------------------------------------------------
    # Year-by-year results
    # --------------------------------------------------------

    print()
    print(
        "================================"
    )
    print(
        "       V23 BY YEAR"
    )
    print(
        "================================"
    )

    yearly = []

    for year, group in test.groupby(
        test["Date"].dt.year
    ):

        selected = (
            group["signal"] == 1
        )

        n = int(
            selected.sum()
        )

        if n:

            reward = (
                group.loc[
                    selected,
                    "future_5d_return"
                ].mean()
            )

            acc = (
                group.loc[
                    selected,
                    "trade_label"
                ].mean()
            )

        else:

            reward = 0.0
            acc = 0.0

        yearly.append({
            "year": year,
            "selected_trades": n,
            "trade_accuracy": acc,
            "avg_reward": reward,
            "samples": len(group),
        })

    yearly = pd.DataFrame(
        yearly
    )

    print(
        yearly.to_string(
            index=False,
            formatters={
                "trade_accuracy":
                    "{:.4f}".format,

                "avg_reward":
                    "{:.4%}".format,
            }
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": features,
        "importance":
            model.feature_importances_,
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
        "       TOP 15 FEATURES"
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


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "############################################################"
    )

    print(
        "#                         V23"
    )

    print(
        "#       NO RAW PRICE-LEVEL FEATURES"
    )

    print(
        "############################################################"
    )

    data, features = load_data()

    # --------------------------------------------------------
    # Development walk-forward
    # --------------------------------------------------------

    development_predictions = (
        walk_forward(
            data,
            features,
            2020,
            2023
        )
    )

    threshold = select_threshold(
        development_predictions
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    final_test(
        data,
        features,
        threshold
    )

    print()
    print(
        "############################################################"
    )

    print(
        "#                         V23 DONE"
    )

    print(
        "############################################################"
    )


if __name__ == "__main__":
    main()