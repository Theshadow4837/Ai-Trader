"""
V28 PHASE 20 — PAPER-TRADING ENGINE

RESEARCH / PAPER-TRADING ANALYSIS ONLY.

NO TRAINING
NO MODEL MODIFICATION
NO BROKER
NO REAL ORDERS

Loads the frozen V28 model read-only.
Uses the existing 85-feature model input.
Uses REAL SPY OHLC prices from data/SPY.csv
for hypothetical execution and valuation.

Signal at bar t
-> execute at bar t+1 OPEN
-> mark-to-market at CLOSE

SPY_return_1d is NEVER used as a price.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from v28_validation import (
    EXPECTED_MODEL_SHA256,
    compare_feature_schema,
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

SPY_OHLC_PATH = Path(
    "data/SPY.csv"
)

OUTPUT_DIR = Path(
    "data/v28_phase20"
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

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005

SLIPPAGE = 0.0005


# ============================================================
# RESEARCH-ONLY COLUMNS
# ============================================================

VALIDATION_EXCLUDED_COLUMNS = {
    "target",
    "future_1d_return",
    "future_3d_return",
    "future_5d_return",
    "future_10d_return",
    "trade_reward",
    "trade_label",
}


# ============================================================
# MODEL HASH
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_frozen_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Frozen model not found:\n"
            f"{MODEL_PATH}"
        )

    actual_hash = sha256_file(
        MODEL_PATH
    )

    print()
    print("=" * 70)
    print("MODEL INTEGRITY")
    print("=" * 70)

    print(
        f"Expected SHA-256: "
        f"{EXPECTED_MODEL_SHA256}"
    )

    print(
        f"Actual SHA-256:   "
        f"{actual_hash}"
    )

    if actual_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "FROZEN MODEL HASH MISMATCH."
        )

    print("MODEL HASH: PASS")


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    if "Date" not in df.columns:
        raise RuntimeError(
            "Dataset must contain Date."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="raise",
    ).dt.normalize()

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
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# PREPARE FEATURE DATA
# ============================================================

def prepare_data(df):

    features = find_features(
        df
    )

    validation_df = df.drop(
        columns=[
            column
            for column in VALIDATION_EXCLUDED_COLUMNS
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
            "Computed feature order does not match "
            "V28 reference schema."
        )

    comparison = compare_feature_schema(
        validation_df,
        REFERENCE_FEATURE_FILE,
    )

    print()
    print("=" * 70)
    print("V28 FEATURE INTEGRITY")
    print("=" * 70)

    print(
        f"Expected features: "
        f"{len(comparison['expected'])}"
    )

    print(
        f"Live features:     "
        f"{len(comparison['live'])}"
    )

    print(
        f"Missing:           "
        f"{comparison['missing'] or 'none'}"
    )

    print(
        f"Extra:             "
        f"{comparison['extra'] or 'none'}"
    )

    print(
        f"Order match:       "
        f"{comparison['order_match']}"
    )

    if len(features) != 85:
        raise RuntimeError(
            f"Expected 85 V28 features, "
            f"found {len(features)}."
        )

    train = df[
        (df["Date"] >= TRAIN_START)
        &
        (df["Date"] <= TRAIN_END)
    ].copy()

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
    ).astype(
        np.float32
    )

    return (
        test,
        test_X,
        features,
    )


# ============================================================
# LOAD AND MERGE REAL SPY OHLC
# ============================================================

def load_and_merge_spy_prices(test):

    if not SPY_OHLC_PATH.exists():
        raise FileNotFoundError(
            f"SPY OHLC file not found:\n"
            f"{SPY_OHLC_PATH}"
        )

    print()
    print("=" * 70)
    print("LOADING REAL SPY OHLC DATA")
    print("=" * 70)

    spy = pd.read_csv(
        SPY_OHLC_PATH
    )

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required
        if column not in spy.columns
    ]

    if missing:
        raise RuntimeError(
            "PHASE 20 ABORTED.\n\n"
            f"SPY OHLC file: {SPY_OHLC_PATH}\n"
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(spy.columns)}"
        )

    spy = spy[
        required
    ].copy()

    test = test.copy()

    test["Date"] = pd.to_datetime(
        test["Date"],
        errors="raise",
    ).dt.normalize()

    spy["Date"] = pd.to_datetime(
        spy["Date"],
        errors="raise",
    ).dt.normalize()

    # --------------------------------------------------------
    # Remove duplicate SPY dates only if they are exact
    # duplicates. Otherwise abort.
    # --------------------------------------------------------

    duplicate_mask = spy["Date"].duplicated(
        keep=False
    )

    if duplicate_mask.any():

        duplicate_rows = spy.loc[
            duplicate_mask
        ].copy()

        duplicate_dates = (
            duplicate_rows["Date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "PHASE 20 ABORTED.\n\n"
            "Duplicate dates found in SPY.csv.\n"
            f"Examples: {duplicate_dates[:10]}"
        )

    # --------------------------------------------------------
    # Convert OHLC to numeric.
    # --------------------------------------------------------

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        spy[column] = pd.to_numeric(
            spy[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Reject missing/non-finite/zero prices.
    # --------------------------------------------------------

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        values = spy[column].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():

            raise RuntimeError(
                "PHASE 20 ABORTED.\n\n"
                f"Non-finite values found in SPY {column}."
            )

        if (values <= 0).any():

            raise RuntimeError(
                "PHASE 20 ABORTED.\n\n"
                f"Non-positive values found in SPY {column}."
            )

    # --------------------------------------------------------
    # OHLC relationship validation.
    #
    # IMPORTANT:
    # Use a small floating-point tolerance.
    #
    # Some adjusted-price datasets can contain extremely
    # small numerical discrepancies.
    #
    # We DO NOT require exact equality.
    # --------------------------------------------------------

    open_values = spy["Open"].to_numpy(
        dtype=np.float64
    )

    high_values = spy["High"].to_numpy(
        dtype=np.float64
    )

    low_values = spy["Low"].to_numpy(
        dtype=np.float64
    )

    close_values = spy["Close"].to_numpy(
        dtype=np.float64
    )

    tolerance = (
        1e-8
        * np.maximum(
            1.0,
            np.maximum(
                np.abs(open_values),
                np.maximum(
                    np.abs(high_values),
                    np.maximum(
                        np.abs(low_values),
                        np.abs(close_values),
                    ),
                ),
            ),
        )
    )

    invalid_high = (
        high_values
        + tolerance
        <
        np.maximum(
            open_values,
            close_values,
        )
    )

    invalid_low = (
        low_values
        - tolerance
        >
        np.minimum(
            open_values,
            close_values,
        )
    )

    invalid_range = (
        high_values
        + tolerance
        <
        low_values
    )

    invalid_mask = (
        invalid_high
        |
        invalid_low
        |
        invalid_range
    )

    if invalid_mask.any():

        bad = spy.loc[
            invalid_mask
        ].copy()

        print()
        print(
            "WARNING: invalid OHLC relationships detected."
        )

        print(
            f"Invalid rows: {len(bad)}"
        )

        print(
            bad[
                [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

        raise RuntimeError(
            "PHASE 20 ABORTED.\n\n"
            "Invalid OHLC relationships found in SPY.csv.\n"
            "The file contains rows where High/Low are "
            "inconsistent with Open/Close."
        )

    # --------------------------------------------------------
    # Merge.
    # --------------------------------------------------------

    spy_prices = spy[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    ].copy()

    test = test.merge(
        spy_prices,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Every holdout row needs OHLC.
    # --------------------------------------------------------

    missing_price_rows = test[
        test[
            [
                "Open",
                "High",
                "Low",
                "Close",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    if not missing_price_rows.empty:

        missing_dates = (
            missing_price_rows["Date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        raise RuntimeError(
            "PHASE 20 ABORTED.\n\n"
            "Some holdout dates have no matching SPY OHLC data.\n"
            f"Missing rows: {len(missing_dates)}\n"
            f"Missing dates: {missing_dates[:20]}"
        )

    # --------------------------------------------------------
    # Final sanity checks after merge.
    # --------------------------------------------------------

    if len(test) == 0:
        raise RuntimeError(
            "PHASE 20 ABORTED.\n\n"
            "Holdout became empty after SPY merge."
        )

    print(
        f"SPY OHLC rows loaded: "
        f"{len(spy)}"
    )

    print(
        f"Holdout rows with OHLC: "
        f"{len(test)}"
    )

    print(
        f"SPY Open range: "
        f"{test['Open'].min():.6f} -> "
        f"{test['Open'].max():.6f}"
    )

    print(
        f"SPY Close range: "
        f"{test['Close'].min():.6f} -> "
        f"{test['Close'].max():.6f}"
    )

    print(
        "SPY OHLC integrity: PASS"
    )

    return test


# ============================================================
# FROZEN MODEL PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    test_X,
):

    predictions = []

    for observation in test_X:

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = int(
            np.asarray(
                action
            ).item()
        )

        predictions.append(
            1
            if action == 1
            else 0
        )

    return np.asarray(
        predictions,
        dtype=np.int8,
    )


# ============================================================
# EXECUTION PRICE
# ============================================================

def get_execution_price(
    row,
):

    return float(
        row["Open"]
    )


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(
    equity_curve,
):

    values = np.asarray(
        equity_curve,
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    peaks = np.maximum.accumulate(
        values
    )

    drawdowns = (
        values / peaks
    ) - 1.0

    return float(
        drawdowns.min()
    )


# ============================================================
# SHARPE
# ============================================================

def sharpe_ratio(
    returns,
):

    values = np.asarray(
        returns,
        dtype=float,
    )

    if len(values) < 2:
        return 0.0

    std = values.std(
        ddof=1
    )

    if std <= 1e-12:
        return 0.0

    return float(
        np.sqrt(252.0)
        * values.mean()
        / std
    )


# ============================================================
# ANNUALIZED RETURN
# ============================================================

def annualized_return(
    initial,
    final,
    days,
):

    years = (
        days / 365.25
    )

    if years <= 0:
        return 0.0

    if final <= 0:
        return -1.0

    return float(
        (
            final / initial
        )
        ** (
            1.0 / years
        )
        - 1.0
    )


# ============================================================
# PAPER TRADING ENGINE
# ============================================================

def run_paper_trading(
    test,
    predictions,
):

    cash = INITIAL_CAPITAL

    shares = 0.0

    position = 0

    entry_date = None
    entry_price = None
    entry_total_cost = None

    trade_id = 0

    trades = []

    daily_records = []

    pending_position = None
    pending_signal_date = None

    equity_curve = []
    daily_returns = []

    for i in range(
        len(test)
    ):

        row = test.iloc[i]

        date = row["Date"]

        close_price = float(
            row["Close"]
        )

        # ----------------------------------------------------
        # Execute pending signal at today's OPEN.
        # ----------------------------------------------------

        if pending_position is not None:

            new_position = int(
                pending_position
            )

            execution_price = (
                get_execution_price(
                    row
                )
            )

            old_position = position

            # ------------------------------------------------
            # FLAT -> LONG
            # ------------------------------------------------

            if (
                old_position == 0
                and new_position == 1
            ):

                trade_id += 1

                fill_price = (
                    execution_price
                    * (
                        1.0
                        + SLIPPAGE
                    )
                )

                gross_shares = (
                    cash
                    /
                    (
                        fill_price
                        *
                        (
                            1.0
                            + TRANSACTION_COST
                        )
                    )
                )

                gross_value = (
                    gross_shares
                    * fill_price
                )

                commission = (
                    gross_value
                    * TRANSACTION_COST
                )

                total_cost = (
                    gross_value
                    + commission
                )

                cash -= total_cost

                shares = gross_shares

                position = 1

                entry_date = date

                entry_price = fill_price

                entry_total_cost = (
                    total_cost
                )

                trades.append({

                    "trade_id":
                        trade_id,

                    "side":
                        "BUY",

                    "signal_date":
                        pending_signal_date,

                    "execution_date":
                        date,

                    "execution_index":
                        i,

                    "execution_price":
                        fill_price,

                    "shares":
                        shares,

                    "gross_value":
                        gross_value,

                    "transaction_cost":
                        commission,

                    "slippage_rate":
                        SLIPPAGE,

                    "position_after":
                        1,
                })

            # ------------------------------------------------
            # LONG -> FLAT
            # ------------------------------------------------

            elif (
                old_position == 1
                and new_position == 0
            ):

                fill_price = (
                    execution_price
                    * (
                        1.0
                        - SLIPPAGE
                    )
                )

                gross_value = (
                    shares
                    * fill_price
                )

                commission = (
                    gross_value
                    * TRANSACTION_COST
                )

                net_proceeds = (
                    gross_value
                    - commission
                )

                cash += net_proceeds

                if (
                    entry_total_cost is not None
                    and entry_total_cost > 0
                ):

                    trade_return = (
                        net_proceeds
                        /
                        entry_total_cost
                    ) - 1.0

                else:

                    trade_return = 0.0

                if entry_date is not None:

                    holding_days = (
                        date
                        - entry_date
                    ).days

                else:

                    holding_days = 0

                trades.append({

                    "trade_id":
                        trade_id,

                    "side":
                        "SELL",

                    "signal_date":
                        pending_signal_date,

                    "execution_date":
                        date,

                    "execution_index":
                        i,

                    "execution_price":
                        fill_price,

                    "shares":
                        shares,

                    "gross_value":
                        gross_value,

                    "transaction_cost":
                        commission,

                    "slippage_rate":
                        SLIPPAGE,

                    "position_after":
                        0,

                    "completed_trade_return":
                        trade_return,

                    "entry_date":
                        entry_date,

                    "entry_price":
                        entry_price,

                    "holding_days":
                        holding_days,
                })

                shares = 0.0
                position = 0

                entry_date = None
                entry_price = None
                entry_total_cost = None

            pending_position = None
            pending_signal_date = None

        # ----------------------------------------------------
        # Mark to current CLOSE.
        # ----------------------------------------------------

        market_value = (
            shares
            * close_price
        )

        equity = (
            cash
            + market_value
        )

        equity_curve.append(
            equity
        )

        if len(equity_curve) == 1:

            daily_return = 0.0

        else:

            previous_equity = (
                equity_curve[-2]
            )

            if previous_equity > 0:

                daily_return = (
                    equity
                    /
                    previous_equity
                    - 1.0
                )

            else:

                daily_return = 0.0

        daily_returns.append(
            daily_return
        )

        daily_records.append({

            "Date":
                date,

            "Open":
                float(row["Open"]),

            "High":
                float(row["High"]),

            "Low":
                float(row["Low"]),

            "Close":
                close_price,

            "cash":
                cash,

            "shares":
                shares,

            "position":
                position,

            "market_value":
                market_value,

            "equity":
                equity,

            "daily_return":
                daily_return,

            "signal":
                int(predictions[i]),

            "executed_position":
                position,
        })

        # ----------------------------------------------------
        # Generate today's signal.
        #
        # Execute on next bar.
        # ----------------------------------------------------

        if i < len(test) - 1:

            desired_position = int(
                predictions[i]
            )

            if (
                desired_position
                != position
            ):

                pending_position = (
                    desired_position
                )

                pending_signal_date = (
                    date
                )

    # --------------------------------------------------------
    # End of dataset.
    # No artificial liquidation.
    # --------------------------------------------------------

    final_row = test.iloc[-1]

    final_close = float(
        final_row["Close"]
    )

    final_market_value = (
        shares
        * final_close
    )

    final_equity = (
        cash
        + final_market_value
    )

    trades_df = pd.DataFrame(
        trades
    )

    daily_df = pd.DataFrame(
        daily_records
    )

    if len(trades_df) > 0:

        completed = trades_df[
            trades_df["side"]
            == "SELL"
        ].copy()

    else:

        completed = pd.DataFrame()

    days = (
        test["Date"].iloc[-1]
        - test["Date"].iloc[0]
    ).days

    total_return = (
        final_equity
        /
        INITIAL_CAPITAL
        - 1.0
    )

    annualized = annualized_return(
        INITIAL_CAPITAL,
        final_equity,
        days,
    )

    max_dd = max_drawdown(
        equity_curve
    )

    sharpe = sharpe_ratio(
        daily_returns
    )

    if len(completed) > 0:

        trade_returns = (
            completed[
                "completed_trade_return"
            ]
            .astype(float)
            .to_numpy()
        )

        winners = (
            trade_returns > 0
        )

        losers = (
            trade_returns < 0
        )

        win_rate = (
            winners.sum()
            /
            len(trade_returns)
        )

        average_trade = (
            trade_returns.mean()
        )

        median_trade = (
            np.median(
                trade_returns
            )
        )

        best_trade = (
            trade_returns.max()
        )

        worst_trade = (
            trade_returns.min()
        )

        average_holding = (
            completed[
                "holding_days"
            ]
            .astype(float)
            .mean()
        )

    else:

        trade_returns = np.asarray(
            [],
            dtype=float,
        )

        winners = np.asarray(
            [],
            dtype=bool,
        )

        losers = np.asarray(
            [],
            dtype=bool,
        )

        win_rate = 0.0
        average_trade = 0.0
        median_trade = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        average_holding = 0.0

    return {

        "final_equity":
            final_equity,

        "return":
            total_return,

        "annualized":
            annualized,

        "max_drawdown":
            max_dd,

        "sharpe":
            sharpe,

        "completed_trades":
            len(completed),

        "winning_trades":
            int(winners.sum()),

        "losing_trades":
            int(losers.sum()),

        "win_rate":
            win_rate,

        "average_trade":
            average_trade,

        "median_trade":
            median_trade,

        "best_trade":
            best_trade,

        "worst_trade":
            worst_trade,

        "average_holding_days":
            average_holding,

        "open_position":
            position,

        "open_shares":
            shares,

        "open_market_value":
            final_market_value,

        "daily":
            daily_df,

        "trades":
            trades_df,
    }


# ============================================================
# YEARLY STATISTICS
# ============================================================

def yearly_statistics(
    daily_df,
):

    if len(daily_df) == 0:
        return pd.DataFrame()

    rows = []

    years = sorted(
        daily_df["Date"]
        .dt.year
        .unique()
    )

    for year in years:

        period = daily_df[
            daily_df["Date"].dt.year
            == year
        ].copy()

        if len(period) < 2:
            continue

        start_equity = float(
            period["equity"].iloc[0]
        )

        end_equity = float(
            period["equity"].iloc[-1]
        )

        returns = (
            period[
                "daily_return"
            ]
            .astype(float)
            .to_numpy()
        )

        year_curve = (
            period[
                "equity"
            ]
            .astype(float)
            .to_numpy()
        )

        rows.append({

            "year":
                int(year),

            "return":
                (
                    end_equity
                    /
                    start_equity
                    - 1.0
                ),

            "max_drawdown":
                max_drawdown(
                    year_curve
                ),

            "sharpe":
                sharpe_ratio(
                    returns
                ),

            "days":
                len(period),

            "time_in_market":
                period[
                    "position"
                ].mean(),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SPY BENCHMARK
# ============================================================

def run_spy_benchmark(
    test,
):

    if "SPY_return_1d" not in test.columns:
        return None

    returns = (
        test[
            "SPY_return_1d"
        ]
        .astype(float)
        .shift(-1)
        .fillna(0.0)
        .to_numpy()
    )

    equity = INITIAL_CAPITAL

    curve = [
        equity
    ]

    daily_returns = []

    for value in returns:

        equity *= (
            1.0 + value
        )

        curve.append(
            equity
        )

        daily_returns.append(
            value
        )

    days = (
        test["Date"].iloc[-1]
        - test["Date"].iloc[0]
    ).days

    return {

        "final":
            equity,

        "return":
            (
                equity
                /
                INITIAL_CAPITAL
                - 1.0
            ),

        "annualized":
            annualized_return(
                INITIAL_CAPITAL,
                equity,
                days,
            ),

        "max_drawdown":
            max_drawdown(
                curve
            ),

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
    print("=" * 70)
    print("V28 PHASE 20 — PAPER-TRADING ENGINE")
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

    verify_frozen_model()

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

    (
        test,
        test_X,
        features,
    ) = prepare_data(
        df
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
    # REAL SPY OHLC
    # --------------------------------------------------------

    test = load_and_merge_spy_prices(
        test
    )

    open_column = "Open"
    close_column = "Close"

    print()
    print(
        f"Execution price: "
        f"{open_column}"
    )

    print(
        f"Mark-to-market price: "
        f"{close_column}"
    )

    print()

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST * 100:.3f}%"
    )

    print(
        f"Slippage: "
        f"{SLIPPAGE * 100:.3f}%"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "GENERATING DETERMINISTIC FROZEN PREDICTIONS"
    )
    print("=" * 70)

    predictions = generate_predictions(
        model,
        test_X,
    )

    print(
        f"Predictions: "
        f"{len(predictions)}"
    )

    print(
        f"Long signals: "
        f"{int(predictions.sum())}"
    )

    print(
        f"Flat signals: "
        f"{len(predictions) - int(predictions.sum())}"
    )

    # --------------------------------------------------------
    # PAPER TRADING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RUNNING HISTORICAL PAPER TRADING"
    )
    print("=" * 70)

    result = run_paper_trading(
        test,
        predictions,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PAPER-TRADING RESULTS")
    print("=" * 70)

    print(
        f"Initial capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final equity: "
        f"${result['final_equity']:,.2f}"
    )

    print(
        f"Return: "
        f"{result['return'] * 100:+.2f}%"
    )

    print(
        f"Annualized: "
        f"{result['annualized'] * 100:+.2f}%"
    )

    print(
        f"Max drawdown: "
        f"{result['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Sharpe: "
        f"{result['sharpe']:.3f}"
    )

    print()

    print(
        f"Completed trades: "
        f"{result['completed_trades']}"
    )

    print(
        f"Winners: "
        f"{result['winning_trades']}"
    )

    print(
        f"Losers: "
        f"{result['losing_trades']}"
    )

    print(
        f"Win rate: "
        f"{result['win_rate'] * 100:.2f}%"
    )

    print(
        f"Average trade: "
        f"{result['average_trade'] * 100:+.3f}%"
    )

    print(
        f"Median trade: "
        f"{result['median_trade'] * 100:+.3f}%"
    )

    print(
        f"Best trade: "
        f"{result['best_trade'] * 100:+.2f}%"
    )

    print(
        f"Worst trade: "
        f"{result['worst_trade'] * 100:+.2f}%"
    )

    print(
        f"Average holding: "
        f"{result['average_holding_days']:.2f} days"
    )

    if result["open_position"]:

        print()
        print(
            "Open position at end of dataset: YES"
        )

        print(
            f"Unrealized market value: "
            f"${result['open_market_value']:,.2f}"
        )

    else:

        print()
        print(
            "Open position at end of dataset: NO"
        )

    # --------------------------------------------------------
    # YEARLY
    # --------------------------------------------------------

    yearly_df = yearly_statistics(
        result["daily"]
    )

    print()
    print("=" * 70)
    print("YEARLY PAPER-TRADING RESULTS")
    print("=" * 70)

    if len(yearly_df) == 0:

        print(
            "No yearly results."
        )

    else:

        for _, row in yearly_df.iterrows():

            print(
                f"{int(row['year'])} | "
                f"Return "
                f"{row['return'] * 100:+.2f}% | "
                f"DD "
                f"{row['max_drawdown'] * 100:.2f}% | "
                f"Sharpe "
                f"{row['sharpe']:.3f} | "
                f"Time "
                f"{row['time_in_market'] * 100:.1f}%"
            )

    # --------------------------------------------------------
    # SPY BENCHMARK
    # --------------------------------------------------------

    spy = run_spy_benchmark(
        test
    )

    print()
    print("=" * 70)
    print("SPY BENCHMARK")
    print("=" * 70)

    if spy is None:

        print(
            "SPY_return_1d not available."
        )

    else:

        print(
            f"SPY Return: "
            f"{spy['return'] * 100:+.2f}%"
        )

        print(
            f"SPY Annualized: "
            f"{spy['annualized'] * 100:+.2f}%"
        )

        print(
            f"SPY Max DD: "
            f"{spy['max_drawdown'] * 100:.2f}%"
        )

        print(
            f"SPY Sharpe: "
            f"{spy['sharpe']:.3f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result["daily"].to_csv(
        OUTPUT_DIR / "daily_equity.csv",
        index=False,
    )

    result["trades"].to_csv(
        OUTPUT_DIR / "trade_ledger.csv",
        index=False,
    )

    yearly_df.to_csv(
        OUTPUT_DIR / "yearly_results.csv",
        index=False,
    )

    predictions_df = pd.DataFrame({

        "Date":
            test["Date"].to_numpy(),

        "prediction":
            predictions,
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "frozen_predictions.csv",
        index=False,
    )

    summary = {

        "initial_capital":
            INITIAL_CAPITAL,

        "final_equity":
            result["final_equity"],

        "return":
            result["return"],

        "annualized":
            result["annualized"],

        "max_drawdown":
            result["max_drawdown"],

        "sharpe":
            result["sharpe"],

        "completed_trades":
            result["completed_trades"],

        "winning_trades":
            result["winning_trades"],

        "losing_trades":
            result["losing_trades"],

        "win_rate":
            result["win_rate"],

        "average_trade":
            result["average_trade"],

        "median_trade":
            result["median_trade"],

        "best_trade":
            result["best_trade"],

        "worst_trade":
            result["worst_trade"],

        "average_holding_days":
            result["average_holding_days"],

        "transaction_cost":
            TRANSACTION_COST,

        "slippage":
            SLIPPAGE,

        "execution_model":
            "signal_at_t_close_execute_at_t_plus_1_open",

        "execution_price":
            "SPY Open",

        "mark_to_market_price":
            "SPY Close",

        "model":
            str(MODEL_PATH),

        "model_sha256":
            sha256_file(
                MODEL_PATH
            ),
    }

    if spy is not None:

        summary.update({

            "spy_return":
                spy["return"],

            "spy_annualized":
                spy["annualized"],

            "spy_max_drawdown":
                spy["max_drawdown"],

            "spy_sharpe":
                spy["sharpe"],
        })

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 20 OUTPUTS")
    print("=" * 70)

    print(
        f"Saved to:\n"
        f"    {OUTPUT_DIR}/"
    )

    print()
    print("Files:")
    print("    summary.csv")
    print("    daily_equity.csv")
    print("    trade_ledger.csv")
    print("    yearly_results.csv")
    print("    frozen_predictions.csv")

    print()
    print("=" * 70)
    print("V28 STATUS")
    print("=" * 70)

    print("    FROZEN")
    print("    NOT TRAINED")
    print("    NOT MODIFIED")
    print("    NO BROKER")
    print("    NO REAL ORDERS")

    print()
    print("=" * 70)
    print("PHASE 20 COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()