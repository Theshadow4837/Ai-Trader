"""
RESEARCH / PAPER-TRADING ANALYSIS ONLY.

This script:

- NEVER trains V28
- NEVER modifies V28
- NEVER connects to a broker
- NEVER places orders
- Evaluates the frozen V28 deterministically

Tests:

1. Transaction-cost sensitivity
2. Slippage sensitivity
3. Year-by-year robustness
4. V28 vs SPY
5. Frozen-model integrity
6. Feature-schema integrity

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
    "data/v28_robustness"
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

EXPECTED_FEATURE_COUNT = 85

# ============================================================
# ROBUSTNESS ASSUMPTIONS
# ============================================================

TRANSACTION_COSTS = [
    0.0005,   # 0.05%
    0.0010,   # 0.10%
    0.0020,   # 0.20%
    0.0030,   # 0.30%
]

SLIPPAGE_LEVELS = [
    0.0000,   # 0.00%
    0.0005,   # 0.05%
    0.0010,   # 0.10%
]

# ============================================================
# FORBIDDEN NON-FEATURE COLUMNS
#
# These are allowed to exist in the raw research dataset,
# but MUST NOT be passed into the model feature matrix.
# ============================================================

FORBIDDEN_COLUMNS = {
    "future_1d_return",
    "future_3d_return",
    "future_5d_return",
    "future_10d_return",
    "target",
    "trade_reward",
    "trade_label",
}

# ============================================================
# METRICS
# ============================================================

def max_drawdown(curve):

    curve = np.asarray(
        curve,
        dtype=float
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
    days,
):

    years = days / 365.25

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
    print("MODEL INTEGRITY")
    print("-" * 60)

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

    return actual


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

        raise ValueError(
            "Dataset must contain Date."
        )

    if "SPY_return_1d" not in df.columns:

        raise ValueError(
            "Dataset must contain SPY_return_1d "
            "for return evaluation."
        )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SANITIZE DATASET
# ============================================================

def sanitize_feature_dataset(df):

    """
    Remove future/target columns from the DataFrame before
    feature-schema validation.

    IMPORTANT:

    We intentionally keep SPY_return_1d because it is the
    benchmark/evaluation return series, NOT a V28 observation
    feature.

    The model feature list is still determined exclusively
    through find_features().
    """

    forbidden_present = [
        column
        for column in df.columns
        if column.lower() in {
            name.lower()
            for name in FORBIDDEN_COLUMNS
        }
    ]

    if forbidden_present:

        print()
        print("DATASET SANITIZATION")
        print("-" * 60)

        print(
            "Removing non-feature columns from "
            "the model-evaluation DataFrame:"
        )

        for column in forbidden_present:

            print(
                f"    - {column}"
            )

        print()
        print(
            "These columns are NOT used as "
            "V28 observations."
        )

    sanitized = df.drop(
        columns=forbidden_present,
        errors="ignore",
    ).copy()

    return sanitized


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_test_data(df):

    # --------------------------------------------------------
    # Remove target/future columns before validation.
    # --------------------------------------------------------

    df = sanitize_feature_dataset(
        df
    )

    # --------------------------------------------------------
    # Find V28 features
    # --------------------------------------------------------

    features = find_features(
        df
    )

    # --------------------------------------------------------
    # Lock schema to V28 reference schema
    # --------------------------------------------------------

    validated = validate_feature_schema(
        df,
        REFERENCE_FEATURE_FILE
    )

    if features != validated:

        raise RuntimeError(
            "Computed feature order does not match "
            "V28 reference schema."
        )

    comparison = compare_feature_schema(
        df,
        REFERENCE_FEATURE_FILE
    )

    print()
    print("V28 FEATURE INTEGRITY")
    print("-" * 60)

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

    if len(features) != EXPECTED_FEATURE_COUNT:

        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} "
            f"V28 features, found {len(features)}."
        )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    train = df[
        (df["Date"] >= TRAIN_START)
        &
        (df["Date"] <= TRAIN_END)
    ].copy()

    # --------------------------------------------------------
    # Holdout data
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
        test_X - mean
    ) / std

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
        features,
    )


# ============================================================
# FUTURE RETURNS
# ============================================================

def get_future_returns(test):

    """
    IMPORTANT:

    We do NOT use future_1d_return here.

    The raw future-return columns have been removed from the
    model-evaluation DataFrame.

    SPY_return_1d represents today's market return, so we
    shift it by -1 to evaluate today's V28 action against
    the NEXT trading day's market return.
    """

    if "SPY_return_1d" not in test.columns:

        raise ValueError(
            "Dataset contains no SPY_return_1d "
            "benchmark return series."
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
# RUN FROZEN MODEL
# ============================================================

def run_model(
    model,
    test,
    test_X,
    transaction_cost,
    slippage,
):

    future_returns = (
        get_future_returns(test)
    )

    equity = INITIAL_CAPITAL

    position = 0

    curve = [
        equity
    ]

    daily_returns = []

    actions = []

    trades = 0

    long_days = 0

    winning_days = 0

    active_returns = []

    for i in range(
        len(test_X) - 1
    ):

        observation = (
            test_X[i]
        )

        # ----------------------------------------------------
        # FROZEN INFERENCE ONLY
        # ----------------------------------------------------

        action, _ = model.predict(
            observation,
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

        actions.append(
            action
        )

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
            equity / INITIAL_CAPITAL - 1.0,

        "annualized":
            annualized_return(
                INITIAL_CAPITAL,
                equity,
                days
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
                / len(actions)
                if actions
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
                np.mean(active_returns)
                if active_returns
                else 0.0
            ),

        "curve":
            curve,

        "returns":
            daily_returns,
    }


# ============================================================
# SPY BENCHMARK
# ============================================================

def run_spy(test):

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

    for r in returns:

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
        test["Date"].iloc[-1]
        - test["Date"].iloc[0]
    ).days

    return {

        "final":
            equity,

        "return":
            equity / INITIAL_CAPITAL - 1.0,

        "annualized":
            annualized_return(
                INITIAL_CAPITAL,
                equity,
                days
            ),

        "max_dd":
            max_drawdown(
                curve
            ),

        "sharpe":
            sharpe_ratio(
                daily_returns
            ),
    }


# ============================================================
# PERIOD ROBUSTNESS
# ============================================================

def run_period(
    model,
    test,
    test_X,
    start_date,
    end_date,
):

    mask = (
        (test["Date"] >= pd.Timestamp(start_date))
        &
        (test["Date"] <= pd.Timestamp(end_date))
    )

    indices = np.flatnonzero(
        mask.to_numpy()
    )

    if len(indices) < 10:

        return None

    period = (
        test
        .iloc[indices]
        .reset_index(drop=True)
    )

    period_X = (
        test_X[indices]
    )

    return run_model(
        model,
        period,
        period_X,
        transaction_cost=0.001,
        slippage=0.0005,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("V28 ROBUSTNESS TEST")
    print("=" * 70)

    print()
    print("RESEARCH / PAPER TRADING ONLY")
    print("NO TRAINING")
    print("NO MODEL MODIFICATION")
    print("NO BROKER")
    print("NO REAL ORDERS")

    # --------------------------------------------------------
    # MODEL INTEGRITY
    # --------------------------------------------------------

    verify_frozen_model()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING FROZEN MODEL")
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
        prepare_test_data(df)
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
    # TRANSACTION COST ROBUSTNESS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("1. TRANSACTION COST ROBUSTNESS")
    print("=" * 70)

    cost_results = []

    for cost in TRANSACTION_COSTS:

        result = run_model(
            model,
            test,
            test_X,
            transaction_cost=cost,
            slippage=0.0,
        )

        cost_results.append({

            "transaction_cost":
                cost,

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

        })

        print(
            f"{cost * 100:.2f}% | "
            f"Return "
            f"{result['return'] * 100:+.2f}% | "
            f"Annual "
            f"{result['annualized'] * 100:+.2f}% | "
            f"DD "
            f"{result['max_dd'] * 100:.2f}% | "
            f"Sharpe "
            f"{result['sharpe']:.3f} | "
            f"Trades "
            f"{result['trades']}"
        )

    cost_df = pd.DataFrame(
        cost_results
    )

    # --------------------------------------------------------
    # SLIPPAGE ROBUSTNESS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2. SLIPPAGE ROBUSTNESS")
    print("=" * 70)

    slippage_results = []

    for slippage in SLIPPAGE_LEVELS:

        result = run_model(
            model,
            test,
            test_X,
            transaction_cost=0.001,
            slippage=slippage,
        )

        slippage_results.append({

            "slippage":
                slippage,

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

        })

        print(
            f"{slippage * 100:.2f}% | "
            f"Return "
            f"{result['return'] * 100:+.2f}% | "
            f"Annual "
            f"{result['annualized'] * 100:+.2f}% | "
            f"DD "
            f"{result['max_dd'] * 100:.2f}% | "
            f"Sharpe "
            f"{result['sharpe']:.3f}"
        )

    slippage_df = pd.DataFrame(
        slippage_results
    )

    # --------------------------------------------------------
    # YEARLY ROBUSTNESS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3. YEARLY ROBUSTNESS")
    print("=" * 70)

    yearly_results = []

    years = sorted(
        test["Date"]
        .dt.year
        .unique()
    )

    for year in years:

        result = run_period(
            model,
            test,
            test_X,
            f"{year}-01-01",
            f"{year}-12-31",
        )

        if result is None:

            continue

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
            f"{result['trades']}"
        )

    yearly_df = pd.DataFrame(
        yearly_results
    )

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("4. V28 VS SPY")
    print("=" * 70)

    baseline = run_model(
        model,
        test,
        test_X,
        transaction_cost=0.0005,
        slippage=0.0,
    )

    spy = run_spy(
        test
    )

    print()
    print(
        f"V28 Return: "
        f"{baseline['return'] * 100:.2f}%"
    )

    print(
        f"SPY Return: "
        f"{spy['return'] * 100:.2f}%"
    )

    print()
    print(
        f"V28 Annualized: "
        f"{baseline['annualized'] * 100:.2f}%"
    )

    print(
        f"SPY Annualized: "
        f"{spy['annualized'] * 100:.2f}%"
    )

    print()
    print(
        f"V28 Sharpe: "
        f"{baseline['sharpe']:.3f}"
    )

    print(
        f"SPY Sharpe: "
        f"{spy['sharpe']:.3f}"
    )

    print()
    print(
        f"V28 Max DD: "
        f"{baseline['max_dd'] * 100:.2f}%"
    )

    print(
        f"SPY Max DD: "
        f"{spy['max_dd'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cost_df.to_csv(
        OUTPUT_DIR
        / "transaction_costs.csv",
        index=False,
    )

    slippage_df.to_csv(
        OUTPUT_DIR
        / "slippage.csv",
        index=False,
    )

    yearly_df.to_csv(
        OUTPUT_DIR
        / "yearly.csv",
        index=False,
    )

    benchmark_df = pd.DataFrame([

        {
            "model":
                "V28",

            "return":
                baseline["return"],

            "annualized":
                baseline["annualized"],

            "sharpe":
                baseline["sharpe"],

            "max_dd":
                baseline["max_dd"],
        },

        {
            "model":
                "SPY",

            "return":
                spy["return"],

            "annualized":
                spy["annualized"],

            "sharpe":
                spy["sharpe"],

            "max_dd":
                spy["max_dd"],
        },

    ])

    benchmark_df.to_csv(
        OUTPUT_DIR
        / "benchmark.csv",
        index=False,
    )

    # --------------------------------------------------------
    # ROBUSTNESS VERDICT
    # --------------------------------------------------------

    worst_cost = (
        cost_df["sharpe"].min()
    )

    worst_slippage = (
        slippage_df["sharpe"].min()
    )

    negative_years = (
        int(
            (yearly_df["return"] < 0)
            .sum()
        )
        if len(yearly_df)
        else 0
    )

    v28_beats_spy = (
        baseline["return"]
        > spy["return"]
    )

    print()
    print("=" * 70)
    print("5. ROBUSTNESS SUMMARY")
    print("=" * 70)

    print(
        f"Worst cost-case Sharpe: "
        f"{worst_cost:.3f}"
    )

    print(
        f"Worst slippage Sharpe: "
        f"{worst_slippage:.3f}"
    )

    print(
        f"Negative years: "
        f"{negative_years}"
    )

    print(
        f"V28 beats SPY: "
        f"{v28_beats_spy}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROBUSTNESS TEST COMPLETE")
    print("=" * 70)

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()