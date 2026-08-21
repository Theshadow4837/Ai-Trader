from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


MODEL_PATH = Path(
    "models/v28/v28_seed_202_FROZEN.zip"
)

DATA_FILE = Path(
    "data/market_features_v14.csv"
)

OUTPUT_DIR = Path(
    "data/v28_phase17"
)

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-29"
TEST_START = "2024-01-01"

INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.0005
SLIPPAGE = 0.0000


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


def load_data():

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    return (
        df
        .sort_values("Date")
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .reset_index(drop=True)
    )


def prepare_data(df):

    features = find_features(df)

    train = df[
        (df["Date"] >= pd.Timestamp(TRAIN_START))
        &
        (df["Date"] <= pd.Timestamp(TRAIN_END))
    ].copy()

    test = df[
        df["Date"] >= pd.Timestamp(TEST_START)
    ].copy()

    train = train.dropna(
        subset=features
    ).reset_index(drop=True)

    test = test.dropna(
        subset=features
    ).reset_index(drop=True)

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

    std[std < 1e-8] = 1.0

    test_X = (
        (test_X - mean)
        / std
    )

    test_X = np.nan_to_num(
        test_X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    test_X = np.clip(
        test_X,
        -10.0,
        10.0,
    ).astype(np.float32)

    return test, test_X, features


def get_future_returns(test):

    if "future_1d_return" in test.columns:

        return (
            test["future_1d_return"]
            .astype(float)
            .to_numpy()
        )

    if "SPY_return_1d" not in test.columns:

        raise RuntimeError(
            "Missing future return source."
        )

    return (
        test["SPY_return_1d"]
        .shift(-1)
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )


def run_diagnosis(
    model,
    test,
    test_X,
):

    future_returns = (
        get_future_returns(test)
    )

    rows = []

    position = 0
    equity = INITIAL_CAPITAL

    entry_date = None
    entry_equity = None
    entry_price_return = None

    for i in range(len(test_X) - 1):

        date = test["Date"].iloc[i]

        action, _ = model.predict(
            test_X[i],
            deterministic=True,
        )

        action = int(
            np.asarray(action).item()
        )

        new_position = (
            1 if action == 1 else 0
        )

        changed = (
            new_position != position
        )

        cost = (
            TRANSACTION_COST + SLIPPAGE
            if changed
            else 0.0
        )

        market_return = float(
            future_returns[i]
        )

        strategy_return = (
            market_return
            if new_position == 1
            else 0.0
        )

        strategy_return -= cost

        previous_equity = equity

        equity *= (
            1.0 + strategy_return
        )

        # ---------------------------------------------
        # ENTRY
        # ---------------------------------------------

        if new_position == 1 and position == 0:

            entry_date = date
            entry_equity = previous_equity
            entry_price_return = market_return

        # ---------------------------------------------
        # EXIT
        # ---------------------------------------------

        exit_date = None
        holding_days = None
        trade_return = None

        if new_position == 0 and position == 1:

            exit_date = date

            if entry_date is not None:

                holding_days = (
                    date - entry_date
                ).days

                trade_return = (
                    equity / entry_equity
                    - 1.0
                )

        rows.append({
            "date": date,
            "action": action,
            "position": new_position,

            "market_return": market_return,
            "strategy_return": strategy_return,

            "transaction_cost": (
                TRANSACTION_COST
                if changed
                else 0.0
            ),

            "slippage": (
                SLIPPAGE
                if changed
                else 0.0
            ),

            "equity": equity,

            "changed_position": changed,

            "entry_date": entry_date,
            "exit_date": exit_date,

            "holding_days": holding_days,
            "trade_return": trade_return,
        })

        position = new_position

    result = pd.DataFrame(rows)

    return result


def add_trade_ids(df):

    df = df.copy()

    df["trade_id"] = (
        df["position"]
        .ne(df["position"].shift())
        .cumsum()
    )

    return df


def analyze_trades(df):

    trades = []

    in_trade = False

    entry_date = None
    entry_equity = None

    for row in df.itertuples():

        if row.position == 1 and not in_trade:

            in_trade = True

            entry_date = row.date
            entry_equity = row.equity

        elif row.position == 0 and in_trade:

            exit_date = row.date

            trade_return = (
                row.equity
                / entry_equity
                - 1.0
            )

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,

                "holding_days": (
                    exit_date - entry_date
                ).days,

                "trade_return": trade_return,

                "win": (
                    trade_return > 0
                ),
            })

            in_trade = False

    return pd.DataFrame(trades)


def monthly_analysis(df):

    x = df.copy()

    x["month"] = (
        x["date"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        x.groupby("month")
        .agg(
            strategy_return=(
                "strategy_return",
                lambda x:
                    np.prod(1 + x) - 1
            ),
            market_return=(
                "market_return",
                lambda x:
                    np.prod(1 + x) - 1
            ),
            trades=(
                "changed_position",
                "sum"
            ),
        )
        .reset_index()
    )

    monthly["vs_spy"] = (
        monthly["strategy_return"]
        - monthly["market_return"]
    )

    return monthly


def consecutive_losses(trades):

    if len(trades) == 0:
        return 0

    longest = 0
    current = 0

    for win in trades["win"]:

        if not win:
            current += 1
            longest = max(
                longest,
                current
            )

        else:
            current = 0

    return longest


def main():

    print()
    print("=" * 70)
    print("V28 PHASE 17 DIAGNOSIS")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH
        )

    model = PPO.load(
        str(MODEL_PATH),
        device="cpu",
    )

    df = load_data()

    test, test_X, features = (
        prepare_data(df)
    )

    print()
    print(f"Features: {len(features)}")
    print(
        f"Test: "
        f"{test['Date'].iloc[0].date()} "
        f"→ "
        f"{test['Date'].iloc[-1].date()}"
    )

    # ======================================================
    # 1. DAILY DECISIONS
    # ======================================================

    daily = run_diagnosis(
        model,
        test,
        test_X,
    )

    # ======================================================
    # 2. TRADE ANALYSIS
    # ======================================================

    trades = analyze_trades(
        daily
    )

    # ======================================================
    # 3. MONTHLY
    # ======================================================

    monthly = monthly_analysis(
        daily
    )

    # ======================================================
    # 4. SUMMARY
    # ======================================================

    total_long_days = int(
        daily["position"].sum()
    )

    total_days = len(daily)

    trades_count = len(trades)

    wins = int(
        trades["win"].sum()
    ) if trades_count else 0

    losses = (
        trades_count - wins
    )

    print()
    print("=" * 70)
    print("POSITION ANALYSIS")
    print("=" * 70)

    print(
        f"Total days:       {total_days}"
    )

    print(
        f"Long days:        {total_long_days}"
    )

    print(
        f"Time in market:   "
        f"{total_long_days / total_days:.2%}"
    )

    print(
        f"Position changes:  "
        f"{daily['changed_position'].sum()}"
    )

    print()
    print("=" * 70)
    print("TRADE ANALYSIS")
    print("=" * 70)

    print(
        f"Trades:            {trades_count}"
    )

    print(
        f"Wins:              {wins}"
    )

    print(
        f"Losses:            {losses}"
    )

    print(
        f"Win rate:          "
        f"{wins / trades_count:.2%}"
        if trades_count
        else "Win rate:          0.00%"
    )

    if trades_count:

        print(
            f"Average trade:     "
            f"{trades['trade_return'].mean():+.2%}"
        )

        print(
            f"Median trade:      "
            f"{trades['trade_return'].median():+.2%}"
        )

        print(
            f"Best trade:        "
            f"{trades['trade_return'].max():+.2%}"
        )

        print(
            f"Worst trade:       "
            f"{trades['trade_return'].min():+.2%}"
        )

        print(
            f"Avg holding days:  "
            f"{trades['holding_days'].mean():.2f}"
        )

        print(
            f"Longest losing streak: "
            f"{consecutive_losses(trades)}"
        )

    # ======================================================
    # 5. SAVE EVERYTHING
    # ======================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    daily.to_csv(
        OUTPUT_DIR / "daily_decisions.csv",
        index=False,
    )

    trades.to_csv(
        OUTPUT_DIR / "trades.csv",
        index=False,
    )

    monthly.to_csv(
        OUTPUT_DIR / "monthly.csv",
        index=False,
    )

    # ======================================================
    # 6. WORST / BEST
    # ======================================================

    if trades_count:

        trades.sort_values(
            "trade_return"
        ).head(20).to_csv(
            OUTPUT_DIR / "worst_20_trades.csv",
            index=False,
        )

        trades.sort_values(
            "trade_return",
            ascending=False
        ).head(20).to_csv(
            OUTPUT_DIR / "best_20_trades.csv",
            index=False,
        )

    # ======================================================
    # 7. SUMMARY JSON
    # ======================================================

    summary = {
        "total_days": total_days,
        "long_days": total_long_days,
        "time_in_market": (
            total_long_days / total_days
        ),
        "position_changes": int(
            daily["changed_position"].sum()
        ),
        "trades": trades_count,
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / trades_count
            if trades_count
            else 0.0
        ),
        "average_trade_return": (
            float(
                trades["trade_return"].mean()
            )
            if trades_count
            else 0.0
        ),
        "median_trade_return": (
            float(
                trades["trade_return"].median()
            )
            if trades_count
            else 0.0
        ),
        "best_trade": (
            float(
                trades["trade_return"].max()
            )
            if trades_count
            else 0.0
        ),
        "worst_trade": (
            float(
                trades["trade_return"].min()
            )
            if trades_count
            else 0.0
        ),
        "average_holding_days": (
            float(
                trades["holding_days"].mean()
            )
            if trades_count
            else 0.0
        ),
        "longest_losing_streak":
            consecutive_losses(trades),
    }

    import json

    with open(
        OUTPUT_DIR / "summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("PHASE 17 COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()
