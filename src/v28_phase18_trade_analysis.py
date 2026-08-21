"""
V28 PHASE 18 — TRADE-LEVEL DIAGNOSIS

RESEARCH / PAPER-TRADING ANALYSIS ONLY.

This script:
- NEVER trains V28
- NEVER modifies V28
- NEVER connects to a broker
- NEVER places orders
- Uses deterministic frozen V28 inference
- Analyzes the untouched 2024+ holdout

Outputs:
- Trade-by-trade CSV
- Yearly trade statistics
- Winning/losing trade statistics
- Holding-period statistics
- Contribution analysis
"""

from pathlib import Path

import hashlib
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from v28_validation import (
    EXPECTED_MODEL_SHA256,
    find_features,
    validate_feature_schema,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "models/v28/v28_seed_202_FROZEN.zip"
)

DATA_FILE = Path(
    "data/market_features_v14.csv"
)

REFERENCE_FEATURE_FILE = Path(
    "data/market_features_v14.csv"
)

OUTPUT_DIR = Path(
    "data/v28_phase18"
)

TRAIN_START = pd.Timestamp(
    "2015-01-01"
)

TRAIN_END = pd.Timestamp(
    "2023-12-29"
)

TEST_START = pd.Timestamp(
    "2024-01-01"
)

TRANSACTION_COST = 0.0005

SLIPPAGE = 0.0

INITIAL_CAPITAL = 10_000.0


# ============================================================
# HASH
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    actual = sha256_file(
        MODEL_PATH
    )

    print()
    print("=" * 70)
    print("MODEL INTEGRITY")
    print("=" * 70)

    print(
        f"Expected SHA-256: {EXPECTED_MODEL_SHA256}"
    )

    print(
        f"Actual SHA-256:   {actual}"
    )

    if actual != EXPECTED_MODEL_SHA256:

        raise RuntimeError(
            "FROZEN MODEL HASH MISMATCH."
        )

    print("MODEL HASH: PASS")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
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
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# PREPARE HOLDOUT
# ============================================================

# ============================================================
# PREPARE HOLDOUT
# ============================================================

def prepare_data(df):

    features = find_features(
        df
    )

    # ========================================================
    # REMOVE RESEARCH-ONLY FUTURE/TARGET COLUMNS BEFORE
    # MODEL FEATURE VALIDATION
    #
    # These columns may exist in the raw dataset for diagnosis,
    # but they MUST NOT participate in V28 feature validation.
    # ========================================================

    validation_excluded_columns = [
        "target",
        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return",
        "trade_reward",
        "trade_label",
    ]

    validation_df = df.drop(
        columns=[
            column
            for column in validation_excluded_columns
            if column in df.columns
        ],
        errors="ignore",
    )

    validated = validate_feature_schema(
        validation_df,
        REFERENCE_FEATURE_FILE,
    )

    if features != validated:

        raise RuntimeError(
            "Feature schema does not match V28."
        )

    if len(features) != 85:

        raise RuntimeError(
            f"Expected 85 features, "
            f"found {len(features)}."
        )

    # ========================================================
    # TRAINING DATA
    # ========================================================

    train = df[
        (df["Date"] >= TRAIN_START)
        &
        (df["Date"] <= TRAIN_END)
    ].copy()

    # ========================================================
    # HOLDOUT DATA
    # ========================================================

    test = df[
        df["Date"] >= TEST_START
    ].copy()

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

    if len(train) == 0:

        raise RuntimeError(
            "No complete training rows."
        )

    if len(test) < 10:

        raise RuntimeError(
            "Not enough holdout rows."
        )

    # ========================================================
    # TRAINING-ONLY NORMALIZATION
    # ========================================================

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
        axis=0,
    )

    std = np.nanstd(
        train_X,
        axis=0,
    )

    std[
        std < 1e-8
    ] = 1.0

    test_X = (
        test_X - mean
    ) / std

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

    return (
        test,
        test_X,
        features,
    )


# ============================================================
# GET NEXT-DAY RETURNS
# ============================================================

def get_next_day_returns(test):

    if "SPY_return_1d" not in test.columns:

        raise RuntimeError(
            "Dataset must contain SPY_return_1d."
        )

    return (
        test["SPY_return_1d"]
        .shift(-1)
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )


# ============================================================
# RUN DIAGNOSTIC
# ============================================================

def analyze_trades(
    model,
    test,
    test_X,
):

    future_returns = (
        get_next_day_returns(test)
    )

    position = 0

    current_trade = None

    trades = []

    equity = INITIAL_CAPITAL

    equity_curve = [
        equity
    ]

    daily_returns = []

    actions = []

    for i in range(
        len(test_X) - 1
    ):

        observation = (
            test_X[i]
        )

        # ----------------------------------------------------
        # FROZEN DETERMINISTIC INFERENCE
        # ----------------------------------------------------

        action, _ = model.predict(
            observation,
            deterministic=True,
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

        date = (
            test["Date"].iloc[i]
        )

        price = (
            float(
                test["Close"].iloc[i]
            )
            if "Close" in test.columns
            else np.nan
        )

        next_return = (
            float(
                future_returns[i]
            )
        )

        # ----------------------------------------------------
        # ENTER LONG
        # ----------------------------------------------------

        if position == 0 and new_position == 1:

            current_trade = {

                "entry_date":
                    date,

                "entry_price":
                    price,

                "entry_index":
                    i,

                "gross_return":
                    0.0,

                "holding_days":
                    0,

                "market_returns":
                    [],
            }

            # Entry cost
            equity *= (
                1.0
                - TRANSACTION_COST
                - SLIPPAGE
            )

            position = 1

        # ----------------------------------------------------
        # ACTIVE LONG
        # ----------------------------------------------------

        if new_position == 1:

            if current_trade is not None:

                current_trade[
                    "market_returns"
                ].append(
                    next_return
                )

                current_trade[
                    "holding_days"
                ] += 1

            strategy_return = (
                next_return
            )

        else:

            strategy_return = 0.0

        # ----------------------------------------------------
        # EXIT LONG
        # ----------------------------------------------------

        if position == 1 and new_position == 0:

            if current_trade is not None:

                returns = np.asarray(
                    current_trade[
                        "market_returns"
                    ],
                    dtype=float,
                )

                if len(returns) > 0:

                    gross_return = float(
                        np.prod(
                            1.0 + returns
                        ) - 1.0
                    )

                else:

                    gross_return = 0.0

                current_trade[
                    "gross_return"
                ] = gross_return

                current_trade[
                    "exit_date"
                ] = date

                current_trade[
                    "exit_price"
                ] = price

                current_trade[
                    "exit_index"
                ] = i

                current_trade[
                    "transaction_cost"
                ] = (
                    TRANSACTION_COST * 2.0
                )

                current_trade[
                    "slippage"
                ] = (
                    SLIPPAGE * 2.0
                )

                net_return = (
                    (
                        1.0
                        + gross_return
                    )
                    * (
                        1.0
                        - TRANSACTION_COST
                        - SLIPPAGE
                    )
                    - 1.0
                )

                current_trade[
                    "net_return"
                ] = net_return

                current_trade[
                    "winner"
                ] = (
                    net_return > 0
                )

                current_trade[
                    "year"
                ] = int(
                    current_trade[
                        "entry_date"
                    ].year
                )

                trades.append(
                    current_trade
                )

                current_trade = None

            # Exit cost
            equity *= (
                1.0
                - TRANSACTION_COST
                - SLIPPAGE
            )

            position = 0

        # ----------------------------------------------------
        # DAILY EQUITY
        # ----------------------------------------------------

        if new_position == 1:

            equity *= (
                1.0 + strategy_return
            )

        daily_returns.append(
            strategy_return
        )

        equity_curve.append(
            equity
        )

        actions.append(
            new_position
        )

    # --------------------------------------------------------
    # CLOSE OPEN TRADE AT END
    # --------------------------------------------------------

    if current_trade is not None:

        returns = np.asarray(
            current_trade[
                "market_returns"
            ],
            dtype=float,
        )

        if len(returns) > 0:

            gross_return = float(
                np.prod(
                    1.0 + returns
                ) - 1.0
            )

        else:

            gross_return = 0.0

        current_trade[
            "gross_return"
        ] = gross_return

        current_trade[
            "exit_date"
        ] = test["Date"].iloc[-1]

        current_trade[
            "exit_price"
        ] = (
            float(
                test["Close"].iloc[-1]
            )
            if "Close" in test.columns
            else np.nan
        )

        current_trade[
            "exit_index"
        ] = len(test) - 1

        current_trade[
            "transaction_cost"
        ] = (
            TRANSACTION_COST * 2.0
        )

        current_trade[
            "slippage"
        ] = (
            SLIPPAGE * 2.0
        )

        net_return = (
            (
                1.0
                + gross_return
            )
            * (
                1.0
                - TRANSACTION_COST
                - SLIPPAGE
            )
            - 1.0
        )

        current_trade[
            "net_return"
        ] = net_return

        current_trade[
            "winner"
        ] = (
            net_return > 0
        )

        current_trade[
            "year"
        ] = int(
            current_trade[
                "entry_date"
            ].year
        )

        trades.append(
            current_trade
        )

    return (
        pd.DataFrame(trades),
        equity_curve,
        daily_returns,
        actions,
    )


# ============================================================
# TRADE STATISTICS
# ============================================================

def print_trade_statistics(
    trades
):

    print()
    print("=" * 70)
    print("TRADE STATISTICS")
    print("=" * 70)

    if trades.empty:

        print("No trades found.")

        return

    winners = trades[
        trades["winner"] == True
    ]

    losers = trades[
        trades["winner"] == False
    ]

    print(
        f"Trades:              {len(trades)}"
    )

    print(
        f"Winners:             {len(winners)}"
    )

    print(
        f"Losers:              {len(losers)}"
    )

    print(
        f"Win rate:            "
        f"{len(winners) / len(trades) * 100:.2f}%"
    )

    print(
        f"Average trade:       "
        f"{trades['net_return'].mean() * 100:+.3f}%"
    )

    print(
        f"Median trade:        "
        f"{trades['net_return'].median() * 100:+.3f}%"
    )

    print(
        f"Best trade:          "
        f"{trades['net_return'].max() * 100:+.2f}%"
    )

    print(
        f"Worst trade:         "
        f"{trades['net_return'].min() * 100:+.2f}%"
    )

    print(
        f"Average holding:     "
        f"{trades['holding_days'].mean():.2f} days"
    )

    print(
        f"Median holding:      "
        f"{trades['holding_days'].median():.2f} days"
    )

    if not winners.empty:

        print(
            f"Average winner:      "
            f"{winners['net_return'].mean() * 100:+.3f}%"
        )

    if not losers.empty:

        print(
            f"Average loser:       "
            f"{losers['net_return'].mean() * 100:+.3f}%"
        )

    if not winners.empty and not losers.empty:

        avg_win = (
            winners["net_return"].mean()
        )

        avg_loss = abs(
            losers["net_return"].mean()
        )

        if avg_loss > 0:

            print(
                f"Win/loss ratio:      "
                f"{avg_win / avg_loss:.3f}"
            )


# ============================================================
# YEARLY ANALYSIS
# ============================================================

def yearly_analysis(
    trades
):

    if trades.empty:

        return pd.DataFrame()

    rows = []

    for year, group in trades.groupby(
        "year"
    ):

        winners = group[
            group["winner"] == True
        ]

        net_returns = (
            group["net_return"]
        )

        rows.append({

            "year":
                int(year),

            "trades":
                len(group),

            "wins":
                len(winners),

            "losses":
                len(group) - len(winners),

            "win_rate":
                len(winners)
                / len(group),

            "average_trade":
                net_returns.mean(),

            "median_trade":
                net_returns.median(),

            "best_trade":
                net_returns.max(),

            "worst_trade":
                net_returns.min(),

            "average_holding_days":
                group[
                    "holding_days"
                ].mean(),

            "sum_trade_returns":
                net_returns.sum(),

        })

    result = pd.DataFrame(
        rows
    ).sort_values(
        "year"
    )

    return result


def print_yearly_analysis(
    yearly
):

    print()
    print("=" * 70)
    print("YEARLY TRADE ANALYSIS")
    print("=" * 70)

    if yearly.empty:

        print("No yearly data.")

        return

    for _, row in yearly.iterrows():

        print(
            f"{int(row['year'])} | "
            f"Trades {int(row['trades']):3d} | "
            f"Win "
            f"{row['win_rate'] * 100:6.2f}% | "
            f"Avg "
            f"{row['average_trade'] * 100:+.3f}% | "
            f"Best "
            f"{row['best_trade'] * 100:+.2f}% | "
            f"Worst "
            f"{row['worst_trade'] * 100:+.2f}% | "
            f"Hold "
            f"{row['average_holding_days']:.2f}d"
        )


# ============================================================
# LOSS STREAK
# ============================================================

def longest_losing_streak(
    trades
):

    if trades.empty:

        return 0

    longest = 0

    current = 0

    for winner in trades[
        "winner"
    ].tolist():

        if not winner:

            current += 1

            longest = max(
                longest,
                current,
            )

        else:

            current = 0

    return longest


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V28 PHASE 18 — TRADE DIAGNOSIS")
    print("=" * 70)

    print()
    print("RESEARCH / PAPER-TRADING ANALYSIS ONLY")
    print("NO TRAINING")
    print("NO MODEL MODIFICATION")
    print("NO BROKER")
    print("NO REAL ORDERS")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    verify_model()

    print()
    print("=" * 70)
    print("LOADING FROZEN V28")
    print("=" * 70)

    model = PPO.load(
        str(MODEL_PATH),
        device="cpu",
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Action space: "
        f"{model.action_space}"
    )

    print(
        f"Observation space: "
        f"{model.observation_space}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = load_data()

    test, test_X, features = (
        prepare_data(df)
    )

    print()
    print(
        f"Holdout: "
        f"{test['Date'].iloc[0].date()} "
        f"→ "
        f"{test['Date'].iloc[-1].date()}"
    )

    print(
        f"Holdout rows: {len(test)}"
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST * 100:.3f}%"
    )

    print(
        f"Slippage: "
        f"{SLIPPAGE * 100:.3f}%"
    )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    trades, equity_curve, daily_returns, actions = (
        analyze_trades(
            model,
            test,
            test_X,
        )
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_trade_statistics(
        trades
    )

    yearly = yearly_analysis(
        trades
    )

    print_yearly_analysis(
        yearly
    )

    print()
    print("=" * 70)
    print("ADDITIONAL DIAGNOSTICS")
    print("=" * 70)

    print(
        f"Longest losing streak: "
        f"{longest_losing_streak(trades)}"
    )

    if not trades.empty:

        total_positive = (
            trades.loc[
                trades["net_return"] > 0,
                "net_return",
            ].sum()
        )

        total_negative = abs(
            trades.loc[
                trades["net_return"] < 0,
                "net_return",
            ].sum()
        )

        if total_negative > 0:

            profit_factor = (
                total_positive
                / total_negative
            )

            print(
                f"Profit factor: "
                f"{profit_factor:.3f}"
            )

        else:

            print(
                "Profit factor: infinite"
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        OUTPUT_DIR
        / "trade_by_trade.csv",
        index=False,
    )

    yearly.to_csv(
        OUTPUT_DIR
        / "yearly_trade_analysis.csv",
        index=False,
    )

    pd.DataFrame({

        "Date":
            test["Date"].iloc[
                :len(actions)
            ].values,

        "action":
            actions,

        "daily_return":
            daily_returns,

        "equity":
            equity_curve[
                1:len(actions) + 1
            ],

    }).to_csv(
        OUTPUT_DIR
        / "daily_diagnostics.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("PHASE 18 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Saved to:\n"
        f"    {OUTPUT_DIR}/"
    )

    print()
    print("V28 STATUS:")
    print("    FROZEN")
    print("    NOT TRAINED")
    print("    NOT MODIFIED")
    print("    NO BROKER")
    print("    NO REAL ORDERS")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
