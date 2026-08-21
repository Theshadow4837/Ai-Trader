"""
V28 PHASE 18 — STABILITY / EXECUTION SENSITIVITY

RESEARCH / PAPER-TRADING ANALYSIS ONLY.

This script:

- NEVER trains V28
- NEVER modifies V28
- NEVER connects to a broker
- NEVER places real orders
- Loads the frozen V28 model read-only
- Uses deterministic predictions
- Uses training-period statistics only for normalization
- Tests hypothetical execution filters
- Measures whether V28's edge survives reduced turnover

V28:
models/v28/v28_seed_202_FROZEN.zip
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

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005
SLIPPAGE = 0.0005


# ============================================================
# MODEL HASH
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_frozen_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Frozen model not found:\n{MODEL_PATH}"
        )

    actual = sha256_file(
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
        f"{actual}"
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
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    if "Date" not in df.columns:

        raise RuntimeError(
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
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(df):

    features = find_features(
        df
    )

    # --------------------------------------------------------
    # Remove research-only target/future columns BEFORE
    # feature-schema validation.
    # --------------------------------------------------------

    validation_excluded = [
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
            for column in validation_excluded
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

    # --------------------------------------------------------
    # TRAINING PERIOD
    # --------------------------------------------------------

    train = df[
        (df["Date"] >= TRAIN_START)
        &
        (df["Date"] <= TRAIN_END)
    ].copy()

    # --------------------------------------------------------
    # HOLDOUT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRAINING-ONLY NORMALIZATION
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
# FUTURE RETURNS
# ============================================================

def get_future_returns(test):

    if "future_1d_return" in test.columns:

        return (
            test[
                "future_1d_return"
            ]
            .astype(float)
            .to_numpy()
        )

    if "SPY_return_1d" not in test.columns:

        raise RuntimeError(
            "Dataset contains neither "
            "future_1d_return nor SPY_return_1d."
        )

    return (
        test[
            "SPY_return_1d"
        ]
        .shift(-1)
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )


# ============================================================
# MODEL PREDICTIONS
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
            1 if action == 1 else 0
        )

    return np.asarray(
        predictions,
        dtype=np.int8,
    )


# ============================================================
# METRICS
# ============================================================

def max_drawdown(curve):

    curve = np.asarray(
        curve,
        dtype=float,
    )

    peak = np.maximum.accumulate(
        curve
    )

    drawdown = (
        curve / peak
    ) - 1.0

    return float(
        drawdown.min()
    )


def sharpe_ratio(returns):

    returns = np.asarray(
        returns,
        dtype=float,
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
        (final / initial)
        ** (1.0 / years)
        - 1.0
    )


# ============================================================
# RUN STRATEGY
# ============================================================

def run_strategy(
    test,
    predictions,
    transaction_cost,
    slippage,
    min_hold_days=0,
):

    future_returns = (
        get_future_returns(test)
    )

    equity = INITIAL_CAPITAL

    position = 0

    hold_days = 0

    curve = [
        equity
    ]

    daily_returns = []

    trades = 0

    long_days = 0

    winning_days = 0

    active_returns = []

    for i in range(
        len(predictions) - 1
    ):

        desired_position = int(
            predictions[i]
        )

        # ----------------------------------------------------
        # Optional minimum holding period.
        #
        # This is ONLY a hypothetical execution filter.
        # The frozen model prediction itself is unchanged.
        # ----------------------------------------------------

        if position == 1:

            hold_days += 1

        if (
            position == 1
            and desired_position == 0
            and hold_days < min_hold_days
        ):

            new_position = 1

        else:

            new_position = (
                desired_position
            )

        changed = (
            new_position
            != position
        )

        cost = 0.0

        if changed:

            trades += 1

            cost += (
                transaction_cost
            )

            cost += (
                slippage
            )

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

                winning_days += 1

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

        curve.append(
            equity
        )

        if changed:

            hold_days = 0

        position = (
            new_position
        )

    days = (
        test["Date"].iloc[-1]
        - test["Date"].iloc[0]
    ).days

    active_count = len(
        active_returns
    )

    return {

        "final":
            equity,

        "return":
            equity
            / INITIAL_CAPITAL
            - 1.0,

        "annualized":
            annualized_return(
                INITIAL_CAPITAL,
                equity,
                days,
            ),

        "max_dd":
            max_drawdown(
                curve
            ),

        "sharpe":
            sharpe_ratio(
                daily_returns
            ),

        "trades":
            trades,

        "long_days":
            long_days,

        "time_in_market":
            (
                long_days
                / len(predictions[:-1])
                if len(predictions) > 1
                else 0.0
            ),

        "win_rate":
            (
                winning_days
                / active_count
                if active_count
                else 0.0
            ),

        "avg_active_return":
            (
                float(
                    np.mean(
                        active_returns
                    )
                )
                if active_returns
                else 0.0
            ),

        "curve":
            curve,

        "returns":
            daily_returns,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V28 PHASE 18 — STABILITY / EXECUTION SENSITIVITY")
    print("=" * 70)

    print()
    print("RESEARCH / PAPER-TRADING ANALYSIS ONLY")
    print("NO TRAINING")
    print("NO MODEL MODIFICATION")
    print("NO BROKER")
    print("NO REAL ORDERS")

    # --------------------------------------------------------
    # MODEL INTEGRITY
    # --------------------------------------------------------

    verify_frozen_model()

    # --------------------------------------------------------
    # LOAD FROZEN MODEL
    # --------------------------------------------------------

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
    ) = prepare_test_data(df)

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
    # GENERATE FROZEN PREDICTIONS ONCE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING DETERMINISTIC FROZEN PREDICTIONS")
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
        f"Long predictions: "
        f"{int(predictions.sum())}"
    )

    print(
        f"Flat predictions: "
        f"{len(predictions) - int(predictions.sum())}"
    )

    # --------------------------------------------------------
    # EXECUTION FILTER TESTS
    # --------------------------------------------------------

    configurations = [

        {
            "name": "Baseline",
            "min_hold_days": 0,
        },

        {
            "name": "MinHold_2D",
            "min_hold_days": 2,
        },

        {
            "name": "MinHold_3D",
            "min_hold_days": 3,
        },

        {
            "name": "MinHold_5D",
            "min_hold_days": 5,
        },

        {
            "name": "MinHold_10D",
            "min_hold_days": 10,
        },

    ]

    results = []

    print()
    print("=" * 70)
    print("EXECUTION SENSITIVITY")
    print("=" * 70)

    for config in configurations:

        result = run_strategy(
            test,
            predictions,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            min_hold_days=config[
                "min_hold_days"
            ],
        )

        results.append({

            "configuration":
                config["name"],

            "min_hold_days":
                config["min_hold_days"],

            "return":
                result["return"],

            "annualized":
                result["annualized"],

            "max_dd":
                result["max_dd"],

            "sharpe":
                result["sharpe"],

            "trades":
                result["trades"],

            "time_in_market":
                result["time_in_market"],

            "win_rate":
                result["win_rate"],

        })

        print(
            f"{config['name']:12s} | "
            f"Return "
            f"{result['return'] * 100:+.2f}% | "
            f"Annual "
            f"{result['annualized'] * 100:+.2f}% | "
            f"DD "
            f"{result['max_dd'] * 100:.2f}% | "
            f"Sharpe "
            f"{result['sharpe']:.3f} | "
            f"Trades "
            f"{result['trades']} | "
            f"Time "
            f"{result['time_in_market'] * 100:.1f}%"
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # YEARLY BASELINE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("YEARLY BASELINE")
    print("=" * 70)

    yearly_results = []

    years = sorted(
        test["Date"]
        .dt.year
        .unique()
    )

    for year in years:

        mask = (
            test["Date"].dt.year
            == year
        )

        indices = np.flatnonzero(
            mask.to_numpy()
        )

        if len(indices) < 10:

            continue

        period = (
            test
            .iloc[indices]
            .reset_index(drop=True)
        )

        period_predictions = (
            predictions[indices]
        )

        result = run_strategy(
            period,
            period_predictions,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            min_hold_days=0,
        )

        yearly_results.append({

            "year":
                int(year),

            "return":
                result["return"],

            "annualized":
                result["annualized"],

            "max_dd":
                result["max_dd"],

            "sharpe":
                result["sharpe"],

            "trades":
                result["trades"],

            "win_rate":
                result["win_rate"],

        })

        print(
            f"{year} | "
            f"Return "
            f"{result['return'] * 100:+.2f}% | "
            f"DD "
            f"{result['max_dd'] * 100:.2f}% | "
            f"Sharpe "
            f"{result['sharpe']:.3f} | "
            f"Trades "
            f"{result['trades']} | "
            f"Win "
            f"{result['win_rate'] * 100:.1f}%"
        )

    yearly_df = pd.DataFrame(
        yearly_results
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "execution_sensitivity.csv",
        index=False,
    )

    yearly_df.to_csv(
        OUTPUT_DIR
        / "yearly_baseline.csv",
        index=False,
    )

    predictions_df = pd.DataFrame({

        "Date":
            test["Date"].to_numpy(),

        "prediction":
            predictions,

    })

    predictions_df.to_csv(
        OUTPUT_DIR
        / "frozen_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    baseline = results_df.iloc[0]

    print()
    print("=" * 70)
    print("PHASE 18 SUMMARY")
    print("=" * 70)

    print(
        f"Baseline return: "
        f"{baseline['return'] * 100:+.2f}%"
    )

    print(
        f"Baseline annualized: "
        f"{baseline['annualized'] * 100:+.2f}%"
    )

    print(
        f"Baseline Sharpe: "
        f"{baseline['sharpe']:.3f}"
    )

    print(
        f"Baseline max drawdown: "
        f"{baseline['max_dd'] * 100:.2f}%"
    )

    print(
        f"Baseline trades: "
        f"{int(baseline['trades'])}"
    )

    if len(results_df) > 1:

        lowest_turnover = (
            results_df
            .sort_values("trades")
            .iloc[0]
        )

        print()
        print(
            f"Lowest-turnover configuration: "
            f"{lowest_turnover['configuration']}"
        )

        print(
            f"Lowest-turnover trades: "
            f"{int(lowest_turnover['trades'])}"
        )

        print(
            f"Lowest-turnover Sharpe: "
            f"{lowest_turnover['sharpe']:.3f}"
        )

    print()
    print(
        f"Results saved to:\n"
        f"    {OUTPUT_DIR}/"
    )

    print()
    print("V28 STATUS:")
    print("    FROZEN")
    print("    NOT TRAINED")
    print("    NOT MODIFIED")
    print("    NO BROKER")
    print("    NO REAL ORDERS")

    print()
    print("=" * 70)
    print("PHASE 18 COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
