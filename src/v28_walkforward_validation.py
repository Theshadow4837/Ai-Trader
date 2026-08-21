"""
============================================================
V28 WALK-FORWARD VALIDATION
============================================================

Research / paper-trading ONLY.

IMPORTANT:
    - Loads the FROZEN V28 model.
    - Does NOT train.
    - Does NOT modify the model.
    - Does NOT use 2024+ for model selection.
    - Uses training-period normalization statistics only.
    - Evaluates the frozen policy across sequential windows.

Purpose:
    Determine whether V28 behaves consistently across
    different historical market regimes.

Model:
    models/v28/v28_seed_202_FROZEN.zip

Data:
    data/market_features_v14.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "models/v28/v28_seed_202_FROZEN.zip"
)

DATA_FILE = Path(
    "data/market_features_v14.csv"
)

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-29"

# Validation begins after the model's main training era.
VALIDATION_START = "2019-01-01"
VALIDATION_END = "2023-12-29"

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005

# Sequential validation windows.
WINDOW_YEARS = 1


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

    if len(equity_curve) == 0:
        return 0.0

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

    if days <= 0:
        return 0.0

    years = days / 365.25

    if years <= 0:
        return 0.0

    return float(
        (final / initial)
        ** (1.0 / years)
        - 1.0
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print(
        "[V28-WF] Loading dataset..."
    )

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
# NORMALIZATION
# ============================================================

def prepare_features(df):

    features = find_features(
        df
    )

    print(
        f"[V28-WF] Features: "
        f"{len(features)}"
    )

    train = df[
        (df["Date"] >= pd.Timestamp(
            TRAIN_START
        ))
        &
        (df["Date"] <= pd.Timestamp(
            TRAIN_END
        ))
    ].copy()

    train = train.dropna(
        subset=features
    ).reset_index(
        drop=True
    )

    train_X = (
        train[features]
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

    return (
        features,
        mean,
        std
    )


def normalize(
    df,
    features,
    mean,
    std
):

    clean = df.dropna(
        subset=features
    ).copy()

    clean = (
        clean
        .sort_values("Date")
        .reset_index(drop=True)
    )

    X = (
        clean[features]
        .astype(np.float32)
        .to_numpy()
    )

    X = (
        (X - mean)
        / std
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X = np.clip(
        X,
        -10.0,
        10.0
    ).astype(np.float32)

    return (
        clean,
        X
    )


# ============================================================
# FUTURE RETURNS
# ============================================================

def get_future_returns(df):

    if "future_1d_return" in df.columns:

        return (
            df[
                "future_1d_return"
            ]
            .astype(float)
            .to_numpy()
        )

    if "SPY_return_1d" not in df.columns:

        raise ValueError(
            "Dataset contains neither "
            "future_1d_return nor SPY_return_1d."
        )

    return (
        df[
            "SPY_return_1d"
        ]
        .shift(-1)
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )


# ============================================================
# RUN WINDOW
# ============================================================

def run_window(
    model,
    window_df,
    X
):

    future_returns = (
        get_future_returns(
            window_df
        )
    )

    equity = INITIAL_CAPITAL

    position = 0

    equity_curve = [
        equity
    ]

    returns = []

    trades = 0

    long_days = 0

    wins = 0

    active_returns = []

    actions = []

    # --------------------------------------------------------
    # Leave final observation alone because it has no
    # following observation to evaluate.
    # --------------------------------------------------------

    for i in range(
        len(X) - 1
    ):

        observation = X[i]

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        action = int(
            np.asarray(
                action
            ).item()
        )

        # V28 is LONG / FLAT.
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

        market_return = float(
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

        returns.append(
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

    days = (
        window_df["Date"].iloc[-1]
        - window_df["Date"].iloc[0]
    ).days

    active_count = len(
        active_returns
    )

    return {
        "final": equity,
        "return": (
            equity
            / INITIAL_CAPITAL
            - 1.0
        ),
        "annualized": annualized_return(
            INITIAL_CAPITAL,
            equity,
            days
        ),
        "max_dd": max_drawdown(
            equity_curve
        ),
        "sharpe": sharpe_ratio(
            returns
        ),
        "trades": trades,
        "long_days": long_days,
        "win_rate": (
            wins / active_count
            if active_count > 0
            else 0.0
        ),
        "avg_active_return": (
            float(
                np.mean(
                    active_returns
                )
            )
            if active_returns
            else 0.0
        ),
        "equity_curve": equity_curve,
        "returns": returns,
        "actions": actions,
    }


# ============================================================
# SPY BASELINE
# ============================================================

def run_spy(window_df):

    returns = (
        window_df[
            "SPY_return_1d"
        ]
        .astype(float)
        .to_numpy()
    )

    equity = INITIAL_CAPITAL

    curve = [
        equity
    ]

    daily_returns = []

    for r in returns[:-1]:

        equity *= (
            1.0 + r
        )

        curve.append(
            equity
        )

        daily_returns.append(
            r
        )

    days = (
        window_df["Date"].iloc[-1]
        - window_df["Date"].iloc[0]
    ).days

    return {
        "final": equity,
        "return": (
            equity
            / INITIAL_CAPITAL
            - 1.0
        ),
        "annualized": annualized_return(
            INITIAL_CAPITAL,
            equity,
            days
        ),
        "max_dd": max_drawdown(
            curve
        ),
        "sharpe": sharpe_ratio(
            daily_returns
        ),
    }


# ============================================================
# BUILD WINDOWS
# ============================================================

def build_windows(df):

    start = pd.Timestamp(
        VALIDATION_START
    )

    end = pd.Timestamp(
        VALIDATION_END
    )

    windows = []

    current = start

    while current <= end:

        window_end = (
            current
            + pd.DateOffset(
                years=WINDOW_YEARS
            )
            - pd.Timedelta(days=1)
        )

        window_end = min(
            window_end,
            end
        )

        windows.append(
            (
                current,
                window_end
            )
        )

        current = (
            window_end
            + pd.Timedelta(days=1)
        )

    return windows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V28 WALK-FORWARD VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD FROZEN MODEL
    # --------------------------------------------------------

    print()
    print(
        "[V28-WF] Loading FROZEN model..."
    )

    model = PPO.load(
        str(MODEL_PATH),
        device="cpu",
    )

    print(
        f"[V28-WF] Model: "
        f"{MODEL_PATH}"
    )

    print(
        f"[V28-WF] Action space: "
        f"{model.action_space}"
    )

    print(
        f"[V28-WF] Observation space: "
        f"{model.observation_space}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = load_data()

    features, mean, std = (
        prepare_features(df)
    )

    # --------------------------------------------------------
    # CHECK OBSERVATION SPACE
    # --------------------------------------------------------

    expected_features = (
        model.observation_space.shape[0]
    )

    if expected_features != len(features):

        raise ValueError(
            "\nObservation mismatch.\n"
            f"Model expects: {expected_features}\n"
            f"Dataset provides: {len(features)}"
        )

    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    windows = build_windows(
        df
    )

    print()
    print(
        f"[V28-WF] Validation period: "
        f"{VALIDATION_START} → "
        f"{VALIDATION_END}"
    )

    print(
        f"[V28-WF] Windows: "
        f"{len(windows)}"
    )

    print(
        "[V28-WF] Model remains FROZEN."
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    results = []

    for index, (
        start,
        end
    ) in enumerate(
        windows,
        start=1
    ):

        print()
        print(
            "-" * 60
        )

        print(
            f"[V28-WF] Window {index}/"
            f"{len(windows)}"
        )

        print(
            f"Period: "
            f"{start.date()} → "
            f"{end.date()}"
        )

        window_raw = df[
            (df["Date"] >= start)
            &
            (df["Date"] <= end)
        ].copy()

        if len(window_raw) < 30:

            print(
                "[V28-WF] Skipping: "
                "not enough rows."
            )

            continue

        window_df, X = normalize(
            window_raw,
            features,
            mean,
            std
        )

        if len(window_df) < 30:

            print(
                "[V28-WF] Skipping after "
                "feature cleanup."
            )

            continue

        result = run_window(
            model,
            window_df,
            X
        )

        spy = run_spy(
            window_df
        )

        excess = (
            result["return"]
            - spy["return"]
        )

        print()
        print(
            f"V28 return: "
            f"{result['return'] * 100:+.2f}%"
        )

        print(
            f"SPY return: "
            f"{spy['return'] * 100:+.2f}%"
        )

        print(
            f"Excess: "
            f"{excess * 100:+.2f}%"
        )

        print(
            f"Max DD: "
            f"{result['max_dd'] * 100:.2f}%"
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
            f"{result['win_rate'] * 100:.2f}%"
        )

        results.append({
            "window": index,
            "start": start.date(),
            "end": end.date(),
            "v28_return":
                result["return"],
            "spy_return":
                spy["return"],
            "excess_return":
                excess,
            "v28_annualized":
                result["annualized"],
            "v28_max_dd":
                result["max_dd"],
            "v28_sharpe":
                result["sharpe"],
            "trades":
                result["trades"],
            "long_days":
                result["long_days"],
            "win_rate":
                result["win_rate"],
            "avg_active_return":
                result["avg_active_return"],
        })

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 60)
    print("V28 WALK-FORWARD RESULTS")
    print("=" * 60)

    if len(results_df) == 0:

        raise RuntimeError(
            "No validation windows completed."
        )

    display_df = (
        results_df.copy()
    )

    display_df[
        "v28_return"
    ] *= 100

    display_df[
        "spy_return"
    ] *= 100

    display_df[
        "excess_return"
    ] *= 100

    display_df[
        "v28_annualized"
    ] *= 100

    display_df[
        "v28_max_dd"
    ] *= 100

    display_df[
        "win_rate"
    ] *= 100

    display_df[
        "avg_active_return"
    ] *= 100

    print(
        display_df.to_string(
            index=False,
            formatters={
                "v28_return":
                    lambda x:
                    f"{x:+.2f}%",
                "spy_return":
                    lambda x:
                    f"{x:+.2f}%",
                "excess_return":
                    lambda x:
                    f"{x:+.2f}%",
                "v28_annualized":
                    lambda x:
                    f"{x:+.2f}%",
                "v28_max_dd":
                    lambda x:
                    f"{x:.2f}%",
                "v28_sharpe":
                    lambda x:
                    f"{x:.3f}",
                "win_rate":
                    lambda x:
                    f"{x:.2f}%",
                "avg_active_return":
                    lambda x:
                    f"{x:.4f}%",
            }
        )
    )

    # --------------------------------------------------------
    # CONSISTENCY
    # --------------------------------------------------------

    mean_return = (
        results_df[
            "v28_return"
        ].mean()
    )

    return_std = (
        results_df[
            "v28_return"
        ].std(
            ddof=1
        )
        if len(results_df) > 1
        else 0.0
    )

    mean_sharpe = (
        results_df[
            "v28_sharpe"
        ].mean()
    )

    positive_windows = int(
        (
            results_df[
                "excess_return"
            ] > 0
        ).sum()
    )

    total_windows = len(
        results_df
    )

    print()
    print("=" * 60)
    print("V28 CONSISTENCY")
    print("=" * 60)

    print(
        f"Mean window return: "
        f"{mean_return * 100:+.2f}%"
    )

    print(
        f"Return std: "
        f"{return_std * 100:.2f}%"
    )

    print(
        f"Mean Sharpe: "
        f"{mean_sharpe:.3f}"
    )

    print(
        f"Positive excess windows: "
        f"{positive_windows}/"
        f"{total_windows}"
    )

    print(
        f"Positive excess rate: "
        f"{positive_windows / total_windows * 100:.1f}%"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = Path(
        "data/v28_walkforward_results.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print()
    print(
        f"Saved results:"
    )

    print(
        f"    {output}"
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("V28 WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 60)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "    The V28 model was NOT modified."
    )

    print(
        "    No training occurred."
    )

    print(
        "    2024+ remains untouched by this test."
    )


if __name__ == "__main__":

    main()
