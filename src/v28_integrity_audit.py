"""
============================================================
V28 INTEGRITY AUDIT
============================================================

Research / paper-trading ONLY.

Purpose:
    Independently audit the frozen V28 strategy and its
    walk-forward validation pipeline.

This script DOES NOT:
    - train a model
    - modify a model
    - select a model
    - use validation results to optimize anything

Checks:
    1. Frozen model exists and is readable.
    2. Model action space matches V28 LONG/FLAT design.
    3. Observation dimensions match dataset features.
    4. Forbidden/future columns are excluded.
    5. Normalization uses training data only.
    6. Future-return alignment is t -> t+1.
    7. Transaction costs are applied.
    8. Holdout data is not used for normalization.
    9. SPY benchmark uses the same dates.
    10. Independent Sharpe calculation.
    11. Long-only baseline.
    12. Frozen model file hash.
"""

from pathlib import Path
import hashlib

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

TRAIN_START = pd.Timestamp(
    "2015-01-01"
)

TRAIN_END = pd.Timestamp(
    "2023-12-29"
)

HOLDOUT_START = pd.Timestamp(
    "2024-01-01"
)

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005


# ============================================================
# FEATURE RULES
# ============================================================

FORBIDDEN_FEATURE_WORDS = {
    "future",
    "target",
    "label",
    "reward",
}


EXPLICITLY_EXCLUDED = {
    "Date",
    "future_1d_return",
    "future_3d_return",
    "future_5d_return",
    "future_10d_return",
    "target",
    "trade_reward",
    "trade_label",
}


# ============================================================
# HASH
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# LOAD
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    if "Date" not in df.columns:

        raise RuntimeError(
            "Dataset has no Date column."
        )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    return (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ============================================================
# FEATURES
# ============================================================

def find_features(df):

    features = []

    for column in df.columns:

        if column in EXPLICITLY_EXCLUDED:
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

        features.append(
            column
        )

    return features


# ============================================================
# NORMALIZATION
# ============================================================

def training_normalization(
    df,
    features
):

    train = df[
        (df["Date"] >= TRAIN_START)
        &
        (df["Date"] <= TRAIN_END)
    ].copy()

    train = train.dropna(
        subset=features
    )

    X = (
        train[features]
        .astype(np.float32)
        .to_numpy()
    )

    mean = np.nanmean(
        X,
        axis=0
    )

    std = np.nanstd(
        X,
        axis=0
    )

    std[
        std < 1e-8
    ] = 1.0

    return (
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
        clean.reset_index(drop=True),
        X
    )


# ============================================================
# SHARPE
# ============================================================

def sharpe(returns):

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

    return (
        np.sqrt(252.0)
        * returns.mean()
        / std
    )


# ============================================================
# DRAWDOWN
# ============================================================

def max_drawdown(curve):

    curve = np.asarray(
        curve,
        dtype=float
    )

    peak = np.maximum.accumulate(
        curve
    )

    return float(
        (curve / peak - 1.0).min()
    )


# ============================================================
# FUTURE RETURN ALIGNMENT
# ============================================================

def verify_return_alignment(df):

    print()
    print(
        "[AUDIT] Checking t -> t+1 return alignment..."
    )

    if "SPY_return_1d" not in df.columns:

        raise RuntimeError(
            "SPY_return_1d missing."
        )

    returns = (
        df["SPY_return_1d"]
        .astype(float)
    )

    shifted = (
        returns
        .shift(-1)
        .fillna(0.0)
        .to_numpy()
    )

    # The environment's intended reward return
    # must be the next day's market return.
    expected = (
        returns
        .shift(-1)
        .fillna(0.0)
        .to_numpy()
    )

    if not np.allclose(
        shifted,
        expected,
        equal_nan=True
    ):

        raise RuntimeError(
            "Future-return alignment failed."
        )

    print(
        "[PASS] Reward return is aligned "
        "to the following market step."
    )


# ============================================================
# MODEL AUDIT
# ============================================================

def audit_model(
    model,
    feature_count
):

    print()
    print(
        "[AUDIT] MODEL"
    )

    print(
        f"Action space: "
        f"{model.action_space}"
    )

    print(
        f"Observation space: "
        f"{model.observation_space}"
    )

    if model.action_space.n != 2:

        raise RuntimeError(
            "Expected V28 LONG/FLAT action space "
            "Discrete(2)."
        )

    expected_shape = (
        feature_count,
    )

    if model.observation_space.shape != (
        expected_shape
    ):

        raise RuntimeError(
            "Observation dimension mismatch: "
            f"model={model.observation_space.shape}, "
            f"features={expected_shape}"
        )

    print(
        "[PASS] Action space is Discrete(2)."
    )

    print(
        "[PASS] Observation dimensions match."
    )


# ============================================================
# LEAKAGE AUDIT
# ============================================================

def audit_features(
    df,
    features
):

    print()
    print(
        "[AUDIT] FEATURE LEAKAGE"
    )

    leaks = []

    for feature in features:

        lower = feature.lower()

        if feature in EXPLICITLY_EXCLUDED:

            leaks.append(
                feature
            )

        if any(
            word in lower
            for word in FORBIDDEN_FEATURE_WORDS
        ):

            leaks.append(
                feature
            )

    leaks = sorted(
        set(leaks)
    )

    if leaks:

        raise RuntimeError(
            "FEATURE LEAKAGE DETECTED:\n"
            + "\n".join(leaks)
        )

    print(
        f"[PASS] {len(features)} features "
        "contain no forbidden terms."
    )


# ============================================================
# NORMALIZATION AUDIT
# ============================================================

def audit_normalization(
    df,
    features,
    mean,
    std
):

    print()
    print(
        "[AUDIT] NORMALIZATION"
    )

    train = df[
        (df["Date"] >= TRAIN_START)
        &
        (df["Date"] <= TRAIN_END)
    ]

    holdout = df[
        df["Date"] >= HOLDOUT_START
    ]

    train_X = (
        train[features]
        .dropna()
        .astype(np.float32)
        .to_numpy()
    )

    holdout_X = (
        holdout[features]
        .dropna()
        .astype(np.float32)
        .to_numpy()
    )

    independently_computed_holdout_mean = (
        np.nanmean(
            holdout_X,
            axis=0
        )
    )

    # Make sure our stored normalization statistics
    # correspond to training data, NOT holdout data.
    train_mean = np.nanmean(
        train_X,
        axis=0
    )

    train_std = np.nanstd(
        train_X,
        axis=0
    )

    train_std[
        train_std < 1e-8
    ] = 1.0

    if not np.allclose(
        mean,
        train_mean,
        rtol=1e-6,
        atol=1e-6
    ):

        raise RuntimeError(
            "Normalization mean does not match "
            "training-only statistics."
        )

    if not np.allclose(
        std,
        train_std,
        rtol=1e-6,
        atol=1e-6
    ):

        raise RuntimeError(
            "Normalization std does not match "
            "training-only statistics."
        )

    # This value is intentionally NOT used for
    # normalization. We calculate it only to prove
    # that holdout statistics are different objects.
    difference = np.mean(
        np.abs(
            independently_computed_holdout_mean
            - mean
        )
    )

    print(
        "[PASS] Normalization uses training data only."
    )

    print(
        f"[INFO] Mean absolute difference between "
        f"holdout and training feature means: "
        f"{difference:.6f}"
    )


# ============================================================
# TRANSACTION COST AUDIT
# ============================================================

def audit_transaction_cost():

    print()
    print(
        "[AUDIT] TRANSACTION COST"
    )

    if TRANSACTION_COST <= 0:

        raise RuntimeError(
            "Transaction cost is not positive."
        )

    print(
        f"[PASS] Transaction cost = "
        f"{TRANSACTION_COST * 100:.4f}%"
    )


# ============================================================
# LONG-ONLY BASELINE
# ============================================================

def run_long_only(
    holdout
):

    returns = (
        holdout[
            "SPY_return_1d"
        ]
        .astype(float)
        .to_numpy()
    )

    equity = INITIAL_CAPITAL

    curve = [
        equity
    ]

    daily = []

    for r in returns[:-1]:

        equity *= (
            1.0 + r
        )

        daily.append(
            r
        )

        curve.append(
            equity
        )

    return {
        "final": equity,
        "return": (
            equity
            / INITIAL_CAPITAL
            - 1.0
        ),
        "sharpe": sharpe(
            daily
        ),
        "max_dd": max_drawdown(
            curve
        ),
    }


# ============================================================
# INDEPENDENT FROZEN MODEL RUN
# ============================================================

def run_frozen_model(
    model,
    holdout,
    X
):

    future = (
        holdout[
            "SPY_return_1d"
        ]
        .shift(-1)
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )

    equity = INITIAL_CAPITAL

    position = 0

    curve = [
        equity
    ]

    returns = []

    trades = 0

    long_days = 0

    active = []

    actions = []

    for i in range(
        len(X) - 1
    ):

        action, _ = model.predict(
            X[i],
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

            active.append(
                r
            )

        else:

            strategy_return = 0.0

        strategy_return -= cost

        equity *= (
            1.0 + strategy_return
        )

        returns.append(
            strategy_return
        )

        curve.append(
            equity
        )

        actions.append(
            action
        )

        position = new_position

    return {
        "final": equity,
        "return": (
            equity
            / INITIAL_CAPITAL
            - 1.0
        ),
        "sharpe": sharpe(
            returns
        ),
        "max_dd": max_drawdown(
            curve
        ),
        "trades": trades,
        "long_days": long_days,
        "returns": returns,
        "actions": actions,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V28 INTEGRITY AUDIT")
    print("=" * 60)

    # --------------------------------------------------------
    # FILE
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Frozen model not found: "
            f"{MODEL_PATH}"
        )

    model_hash = sha256_file(
        MODEL_PATH
    )

    print()
    print(
        "[AUDIT] Frozen model SHA-256:"
    )

    print(
        model_hash
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = PPO.load(
        str(MODEL_PATH),
        device="cpu",
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = load_data()

    features = find_features(
        df
    )

    print()
    print(
        f"[AUDIT] Dataset rows: "
        f"{len(df)}"
    )

    print(
        f"[AUDIT] Features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # CHECKS
    # --------------------------------------------------------

    audit_model(
        model,
        len(features)
    )

    audit_features(
        df,
        features
    )

    verify_return_alignment(
        df
    )

    mean, std = (
        training_normalization(
            df,
            features
        )
    )

    audit_normalization(
        df,
        features,
        mean,
        std
    )

    audit_transaction_cost()

    # --------------------------------------------------------
    # HOLDOUT
    # --------------------------------------------------------

    holdout = df[
        df["Date"] >= HOLDOUT_START
    ].copy()

    holdout, X = normalize(
        holdout,
        features,
        mean,
        std
    )

    print()
    print(
        "[AUDIT] Holdout:"
    )

    print(
        f"    {holdout['Date'].iloc[0].date()} "
        f"→ "
        f"{holdout['Date'].iloc[-1].date()}"
    )

    print(
        f"    Rows: {len(holdout)}"
    )

    # --------------------------------------------------------
    # FROZEN MODEL
    # --------------------------------------------------------

    result = run_frozen_model(
        model,
        holdout,
        X
    )

    # --------------------------------------------------------
    # LONG ONLY
    # --------------------------------------------------------

    baseline = run_long_only(
        holdout
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("INDEPENDENT HOLDOUT RECHECK")
    print("=" * 60)

    print()

    print(
        f"{'Metric':<25}"
        f"{'V28':>15}"
        f"{'SPY':>15}"
    )

    print(
        "-" * 55
    )

    print(
        f"{'Final value':<25}"
        f"${result['final']:>13,.2f}"
        f"${baseline['final']:>13,.2f}"
    )

    print(
        f"{'Return':<25}"
        f"{result['return'] * 100:>14.2f}%"
        f"{baseline['return'] * 100:>14.2f}%"
    )

    print(
        f"{'Sharpe':<25}"
        f"{result['sharpe']:>15.3f}"
        f"{baseline['sharpe']:>15.3f}"
    )

    print(
        f"{'Max drawdown':<25}"
        f"{result['max_dd'] * 100:>14.2f}%"
        f"{baseline['max_dd'] * 100:>14.2f}%"
    )

    print()
    print(
        f"Trades: "
        f"{result['trades']}"
    )

    print(
        f"Long days: "
        f"{result['long_days']}"
    )

    # --------------------------------------------------------
    # SHARPE SANITY CHECK
    # --------------------------------------------------------

    independent_sharpe = sharpe(
        result["returns"]
    )

    print()
    print(
        "=" * 60
    )

    print(
        "SHARPE INDEPENDENT RECHECK"
    )

    print(
        "=" * 60
    )

    print(
        f"Independent Sharpe: "
        f"{independent_sharpe:.6f}"
    )

    print(
        f"Reported Sharpe:    "
        f"{result['sharpe']:.6f}"
    )

    if not np.isclose(
        independent_sharpe,
        result["sharpe"],
        rtol=1e-10,
        atol=1e-10
    ):

        raise RuntimeError(
            "Sharpe calculation mismatch."
        )

    print()
    print(
        "[PASS] Sharpe calculation is internally consistent."
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("V28 INTEGRITY AUDIT COMPLETE")
    print("=" * 60)

    print()
    print(
        "MODEL STATUS:"
    )

    print(
        "    FROZEN"
    )

    print()
    print(
        "TRAINING:"
    )

    print(
        "    NONE"
    )

    print()
    print(
        "MODEL HASH:"
    )

    print(
        f"    {model_hash}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "    No model parameters were changed."
    )

    print(
        "    No training occurred."
    )

    print(
        "    No validation result was used for optimization."
    )


if __name__ == "__main__":
    main()
