from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


MODEL_PATH = Path("models/v26_ppo.zip")
DATA_FILE = Path("data/market_features_v14.csv")

TEST_START = "2024-01-01"
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.0005


def max_drawdown(equity):

    equity = np.asarray(equity, dtype=float)

    peak = np.maximum.accumulate(equity)

    drawdown = (
        equity / peak
    ) - 1.0

    return float(drawdown.min())


def sharpe_ratio(returns):

    returns = np.asarray(
        returns,
        dtype=float
    )

    if len(returns) < 2:
        return 0.0

    std = returns.std(ddof=1)

    if std <= 1e-12:
        return 0.0

    return float(
        np.sqrt(252.0)
        * returns.mean()
        / std
    )


def main():

    print()
    print("=" * 60)
    print("V26 HOLDOUT EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("[V26] Loading model...")

    model = PPO.load(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        "[V26] Loading holdout data..."
    )

    df = pd.read_csv(
        DATA_FILE
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # HOLDOUT
    # --------------------------------------------------------

    df = df[
        df["Date"] >= pd.Timestamp(
            TEST_START
        )
    ].copy()

    if len(df) < 2:

        raise ValueError(
            "Not enough holdout data."
        )

    print()
    print(
        f"[V26] Holdout rows: "
        f"{len(df)}"
    )

    print(
        f"[V26] Holdout period: "
        f"{df['Date'].iloc[0].date()} "
        f"→ "
        f"{df['Date'].iloc[-1].date()}"
    )

    # --------------------------------------------------------
    # SAME FEATURES AS TRAINING
    # --------------------------------------------------------

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
            for word in {
                "future",
                "target",
                "label",
                "reward",
            }
        ):
            continue

        if lower in {
            "date",
            "datetime",
            "timestamp",
        }:
            continue

        features.append(column)

    # --------------------------------------------------------
    # VALID ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=features
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # NORMALIZATION
    #
    # IMPORTANT:
    # These statistics reproduce the V26 training
    # preprocessing from the 2015-2023 training dataset.
    # --------------------------------------------------------

    train = pd.read_csv(
        DATA_FILE
    )

    train["Date"] = pd.to_datetime(
        train["Date"]
    )

    train = train[
        (train["Date"] >= pd.Timestamp("2015-01-01"))
        &
        (train["Date"] <= pd.Timestamp("2023-12-29"))
    ].copy()

    train = train.dropna(
        subset=features
    )

    train_X = (
        train[features]
        .astype(np.float32)
        .to_numpy()
    )

    test_X = (
        df[features]
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

    # --------------------------------------------------------
    # FUTURE RETURNS
    #
    # Used ONLY after the model chooses an action.
    # --------------------------------------------------------

    if "future_1d_return" in df.columns:

        future_returns = (
            df["future_1d_return"]
            .astype(float)
            .to_numpy()
        )

    else:

        future_returns = (
            df["SPY_return_1d"]
            .shift(-1)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

    # --------------------------------------------------------
    # SIMULATION
    # --------------------------------------------------------

    equity = INITIAL_CAPITAL

    position = 0

    equity_curve = [
        equity
    ]

    daily_returns = []

    actions = []

    trade_count = 0

    wins = 0

    active_days = 0

    active_returns = []

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
            np.asarray(action).item()
        )

        # 0 = FLAT
        # 1 = LONG

        new_position = (
            1
            if action == 1
            else 0
        )

        position_changed = (
            new_position
            != position
        )

        cost = (
            TRANSACTION_COST
            if position_changed
            else 0.0
        )

        if position_changed:

            trade_count += 1

        market_return = (
            future_returns[i]
        )

        if new_position == 1:

            strategy_return = (
                market_return
            )

            active_days += 1

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

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    final_value = equity

    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1.0
    )

    years = (
        (
            df["Date"].iloc[-1]
            - df["Date"].iloc[0]
        ).days
        / 365.25
    )

    if years > 0:

        annualized = (
            (final_value / INITIAL_CAPITAL)
            ** (1.0 / years)
            - 1.0
        )

    else:

        annualized = 0.0

    drawdown = max_drawdown(
        equity_curve
    )

    sharpe = sharpe_ratio(
        daily_returns
    )

    if active_days > 0:

        win_rate = (
            wins
            / active_days
        )

        average_active_return = (
            np.mean(
                active_returns
            )
        )

    else:

        win_rate = 0.0
        average_active_return = 0.0

    actions = np.asarray(
        actions
    )

    flat_count = int(
        np.sum(actions == 0)
    )

    long_count = int(
        np.sum(actions == 1)
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("V26 HOLDOUT RESULTS")
    print("=" * 60)

    print(
        f"Final value: "
        f"${final_value:,.2f}"
    )

    print(
        f"Return: "
        f"{total_return * 100:.2f}%"
    )

    print(
        f"Annualized: "
        f"{annualized * 100:.2f}%"
    )

    print(
        f"Max drawdown: "
        f"{drawdown * 100:.2f}%"
    )

    print(
        f"Sharpe: "
        f"{sharpe:.3f}"
    )

    print(
        f"Trades: "
        f"{trade_count}"
    )

    print(
        f"Win rate: "
        f"{win_rate * 100:.2f}%"
    )

    print(
        f"Average active return: "
        f"{average_active_return * 100:.4f}%"
    )

    print()
    print("=" * 60)
    print("ACTION DISTRIBUTION")
    print("=" * 60)

    total_actions = (
        flat_count
        + long_count
    )

    print(
        f"FLAT: {flat_count}"
    )

    print(
        f"LONG: {long_count}"
    )

    if total_actions > 0:

        print(
            f"FLAT: "
            f"{flat_count / total_actions * 100:.2f}%"
        )

        print(
            f"LONG: "
            f"{long_count / total_actions * 100:.2f}%"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
