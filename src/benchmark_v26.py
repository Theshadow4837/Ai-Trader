"""
============================================================
V26 SPY BENCHMARK
============================================================

Research / paper-trading analysis ONLY.

Compares:
    1. V26 LONG / FLAT strategy
    2. SPY buy-and-hold

Same:
    - Starting capital
    - Holdout dates
    - Market data
    - Return period

V26 model:
    models/v26_ppo.zip

Holdout:
    2024-01-01 onward

No training occurs in this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "models/v26_ppo.zip"
)

DATA_FILE = Path(
    "data/market_features_v14.csv"
)

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-29"

TEST_START = "2024-01-01"

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005


# ============================================================
# FEATURES
# ============================================================

FORBIDDEN_FEATURE_WORDS = {
    "future",
    "target",
    "label",
    "reward",
}


def find_features(df):

    excluded = {
        "Date",
        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return",
        "target",
        "trade_reward",
        "trade_label",
    }

    features = []

    for column in df.columns:

        if column in excluded:
            continue

        lower = column.lower()

        if any(
            word in lower
            for word in FORBIDDEN_FEATURE_WORDS
        ):
            continue

        if lower in {
            "date",
            "datetime",
            "timestamp",
        }:
            continue

        features.append(column)

    return features


# ============================================================
# METRICS
# ============================================================

def max_drawdown(equity_curve):

    equity_curve = np.asarray(
        equity_curve,
        dtype=float
    )

    peak = np.maximum.accumulate(
        equity_curve
    )

    drawdowns = (
        equity_curve / peak
    ) - 1.0

    return float(
        drawdowns.min()
    )


def sharpe_ratio(returns):

    returns = np.asarray(
        returns,
        dtype=float
    )

    if len(returns) < 2:
        return 0.0

    std = returns.std(
        ddof=1
    )

    if std <= 1e-12:
        return 0.0

    return float(
        np.sqrt(252.0)
        * returns.mean()
        / std
    )


def annualized_return(
    initial,
    final,
    days
):

    years = days / 365.25

    if years <= 0:
        return 0.0

    return (
        (final / initial)
        ** (1.0 / years)
        - 1.0
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    if "Date" not in df.columns:

        raise ValueError(
            "Dataset must contain Date."
        )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# BUILD NORMALIZED FEATURES
# ============================================================

def prepare_features(df):

    features = find_features(
        df
    )

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    train = df[
        (df["Date"] >= pd.Timestamp(
            TRAIN_START
        ))
        &
        (df["Date"] <= pd.Timestamp(
            TRAIN_END
        ))
    ].copy()

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    test = df[
        df["Date"] >= pd.Timestamp(
            TEST_START
        )
    ].copy()

    # --------------------------------------------------------
    # Remove rows that cannot produce observations.
    # --------------------------------------------------------

    train = train.dropna(
        subset=features
    ).reset_index(
        drop=True
    )

    test = test.dropna(
        subset=features
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # TRAINING NORMALIZATION STATISTICS
    #
    # IMPORTANT:
    # Never calculate normalization statistics from
    # holdout data.
    # --------------------------------------------------------

    train_X = (
        train[features]
        .astype(np.float32)
        .to_numpy()
    )

    test_X = (
        test[features]
        .astype(np.float32)
        .to_numpy()
    )

    mean = np.nanmean(
        train_X,
        axis=0
    )

    std = np.nanstd(
        train_X,
        axis=0
    )

    std[
        std < 1e-8
    ] = 1.0

    test_X = (
        (test_X - mean)
        / std
    )

    test_X = np.nan_to_num(
        test_X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    test_X = np.clip(
        test_X,
        -10.0,
        10.0
    ).astype(np.float32)

    return (
        test,
        test_X,
        features
    )


# ============================================================
# V26 SIMULATION
# ============================================================

def run_v26(
    model,
    test,
    test_X
):

    equity = INITIAL_CAPITAL

    position = 0

    equity_curve = [
        equity
    ]

    daily_returns = []

    actions = []

    trade_count = 0

    long_days = 0

    wins = 0

    active_returns = []

    # --------------------------------------------------------
    # Use next-day SPY return.
    # --------------------------------------------------------

    if "future_1d_return" in test.columns:

        future_returns = (
            test[
                "future_1d_return"
            ]
            .astype(float)
            .to_numpy()
        )

    else:

        future_returns = (
            test[
                "SPY_return_1d"
            ]
            .shift(-1)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

    # --------------------------------------------------------
    # MODEL LOOP
    # --------------------------------------------------------

    for i in range(
        len(test_X) - 1
    ):

        observation = (
            test_X[i]
        )

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        action = int(
            np.asarray(
                action
            ).item()
        )

        # ----------------------------------------------------
        # V26 ACTIONS
        #
        # 0 = FLAT
        # 1 = LONG
        # ----------------------------------------------------

        new_position = (
            1
            if action == 1
            else 0
        )

        changed = (
            new_position
            != position
        )

        cost = (
            TRANSACTION_COST
            if changed
            else 0.0
        )

        if changed:
            trade_count += 1

        market_return = (
            future_returns[i]
        )

        if new_position == 1:

            strategy_return = (
                market_return
            )

            long_days += 1

            active_returns.append(
                market_return
            )

            if market_return > 0:
                wins += 1

        else:

            strategy_return = 0.0

        strategy_return -= cost

        equity *= (
            1.0
            + strategy_return
        )

        daily_returns.append(
            strategy_return
        )

        equity_curve.append(
            equity
        )

        actions.append(
            action
        )

        position = (
            new_position
        )

    return {
        "final": equity,
        "equity_curve": equity_curve,
        "returns": daily_returns,
        "actions": actions,
        "trades": trade_count,
        "long_days": long_days,
        "wins": wins,
        "active_returns": active_returns,
    }


# ============================================================
# SPY BUY AND HOLD
# ============================================================

def run_spy(
    test
):

    # --------------------------------------------------------
    # SPY daily returns already exist in the dataset.
    # --------------------------------------------------------

    returns = (
        test[
            "SPY_return_1d"
        ]
        .astype(float)
        .to_numpy()
    )

    equity = INITIAL_CAPITAL

    equity_curve = [
        equity
    ]

    daily_returns = []

    for r in returns:

        equity *= (
            1.0 + r
        )

        daily_returns.append(
            r
        )

        equity_curve.append(
            equity
        )

    return {
        "final": equity,
        "equity_curve": equity_curve,
        "returns": daily_returns,
    }


# ============================================================
# YEARLY METRICS
# ============================================================

def yearly_v26(
    model,
    df
):

    years = sorted(
        df["Date"]
        .dt.year
        .unique()
    )

    results = []

    for year in years:

        if year < 2024:
            continue

        yearly = df[
            df["Date"].dt.year == year
        ].copy()

        if len(yearly) < 2:
            continue

        # ----------------------------------------------------
        # Reuse normalization from training.
        # ----------------------------------------------------

        features = find_features(
            df
        )

        train = df[
            (df["Date"] >= pd.Timestamp(
                TRAIN_START
            ))
            &
            (df["Date"] <= pd.Timestamp(
                TRAIN_END
            ))
        ].dropna(
            subset=features
        )

        yearly = yearly.dropna(
            subset=features
        )

        if len(yearly) < 2:
            continue

        train_X = (
            train[features]
            .astype(np.float32)
            .to_numpy()
        )

        year_X = (
            yearly[features]
            .astype(np.float32)
            .to_numpy()
        )

        mean = np.nanmean(
            train_X,
            axis=0
        )

        std = np.nanstd(
            train_X,
            axis=0
        )

        std[
            std < 1e-8
        ] = 1.0

        year_X = (
            (year_X - mean)
            / std
        )

        year_X = np.nan_to_num(
            year_X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        year_X = np.clip(
            year_X,
            -10.0,
            10.0
        ).astype(np.float32)

        if "future_1d_return" in yearly.columns:

            future = (
                yearly[
                    "future_1d_return"
                ]
                .astype(float)
                .to_numpy()
            )

        else:

            future = (
                yearly[
                    "SPY_return_1d"
                ]
                .shift(-1)
                .fillna(0.0)
                .astype(float)
                .to_numpy()
            )

        equity = INITIAL_CAPITAL

        position = 0

        returns = []

        curve = [
            equity
        ]

        trades = 0

        long_days = 0

        wins = 0

        for i in range(
            len(year_X) - 1
        ):

            action, _ = model.predict(
                year_X[i],
                deterministic=True
            )

            action = int(
                np.asarray(
                    action
                ).item()
            )

            new_position = (
                1
                if action == 1
                else 0
            )

            changed = (
                new_position
                != position
            )

            cost = (
                TRANSACTION_COST
                if changed
                else 0.0
            )

            if changed:
                trades += 1

            r = future[i]

            if new_position == 1:

                strategy_return = r

                long_days += 1

                if r > 0:
                    wins += 1

            else:

                strategy_return = 0.0

            strategy_return -= cost

            equity *= (
                1.0
                + strategy_return
            )

            returns.append(
                strategy_return
            )

            curve.append(
                equity
            )

            position = (
                new_position
            )

        results.append({
            "year": int(year),
            "v26_return": (
                equity
                / INITIAL_CAPITAL
                - 1.0
            ),
            "v26_max_dd": max_drawdown(
                curve
            ),
            "v26_sharpe": sharpe_ratio(
                returns
            ),
            "trades": trades,
            "long_days": long_days,
            "win_rate": (
                wins / long_days
                if long_days > 0
                else 0.0
            ),
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V26 vs SPY BENCHMARK")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print(
        "[BENCHMARK] Loading V26..."
    )

    model = PPO.load(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = load_data()

    test, test_X, features = (
        prepare_features(df)
    )

    print()
    print(
        f"Holdout: "
        f"{test['Date'].iloc[0].date()} "
        f"→ "
        f"{test['Date'].iloc[-1].date()}"
    )

    print(
        f"Holdout rows: "
        f"{len(test)}"
    )

    print(
        f"Features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # V26
    # --------------------------------------------------------

    print()
    print(
        "[BENCHMARK] Running V26..."
    )

    v26 = run_v26(
        model,
        test,
        test_X
    )

    # --------------------------------------------------------
    # SPY
    # --------------------------------------------------------

    print(
        "[BENCHMARK] Running SPY..."
    )

    spy = run_spy(
        test
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    days = (
        test["Date"].iloc[-1]
        - test["Date"].iloc[0]
    ).days

    v26_return = (
        v26["final"]
        / INITIAL_CAPITAL
        - 1.0
    )

    spy_return = (
        spy["final"]
        / INITIAL_CAPITAL
        - 1.0
    )

    excess_return = (
        v26_return
        - spy_return
    )

    v26_annualized = annualized_return(
        INITIAL_CAPITAL,
        v26["final"],
        days
    )

    spy_annualized = annualized_return(
        INITIAL_CAPITAL,
        spy["final"],
        days
    )

    v26_dd = max_drawdown(
        v26["equity_curve"]
    )

    spy_dd = max_drawdown(
        spy["equity_curve"]
    )

    v26_sharpe = sharpe_ratio(
        v26["returns"]
    )

    spy_sharpe = sharpe_ratio(
        spy["returns"]
    )

    active_returns = (
        v26["active_returns"]
    )

    if active_returns:

        win_rate = (
            v26["wins"]
            / len(active_returns)
        )

        avg_active = np.mean(
            active_returns
        )

    else:

        win_rate = 0.0
        avg_active = 0.0

    total_days = len(
        v26["actions"]
    )

    time_in_market = (
        v26["long_days"]
        / total_days
        if total_days > 0
        else 0.0
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print()
    print(
        f"{'Metric':<25}"
        f"{'V26':>15}"
        f"{'SPY':>15}"
    )

    print("-" * 55)

    print(
        f"{'Final value':<25}"
        f"${v26['final']:>13,.2f}"
        f"${spy['final']:>13,.2f}"
    )

    print(
        f"{'Return':<25}"
        f"{v26_return * 100:>14.2f}%"
        f"{spy_return * 100:>14.2f}%"
    )

    print(
        f"{'Annualized':<25}"
        f"{v26_annualized * 100:>14.2f}%"
        f"{spy_annualized * 100:>14.2f}%"
    )

    print(
        f"{'Max drawdown':<25}"
        f"{v26_dd * 100:>14.2f}%"
        f"{spy_dd * 100:>14.2f}%"
    )

    print(
        f"{'Sharpe':<25}"
        f"{v26_sharpe:>15.3f}"
        f"{spy_sharpe:>15.3f}"
    )

    print()
    print(
        f"V26 excess return: "
        f"{excess_return * 100:+.2f}%"
    )

    print(
        f"V26 trades: "
        f"{v26['trades']}"
    )

    print(
        f"V26 time in market: "
        f"{time_in_market * 100:.2f}%"
    )

    print(
        f"V26 win rate: "
        f"{win_rate * 100:.2f}%"
    )

    print(
        f"V26 average active return: "
        f"{avg_active * 100:.4f}%"
    )

    # --------------------------------------------------------
    # YEARLY
    # --------------------------------------------------------

    yearly = yearly_v26(
        model,
        df
    )

    print()
    print("=" * 60)
    print("V26 BY YEAR")
    print("=" * 60)

    if len(yearly) > 0:

        print(
            yearly.to_string(
                index=False,
                formatters={
                    "v26_return":
                        lambda x:
                        f"{x * 100:.2f}%",
                    "v26_max_dd":
                        lambda x:
                        f"{x * 100:.2f}%",
                    "v26_sharpe":
                        lambda x:
                        f"{x:.3f}",
                    "win_rate":
                        lambda x:
                        f"{x * 100:.2f}%",
                }
            )
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = Path(
        "data/v26_benchmark_yearly.csv"
    )

    yearly.to_csv(
        output,
        index=False
    )

    print()
    print(
        f"Saved yearly results: "
        f"{output}"
    )

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()
