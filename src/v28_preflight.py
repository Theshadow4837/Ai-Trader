"""
V28 production preflight validation.

This script performs read-only safety checks before
the paper-trading pipeline runs.

It does NOT:
- train the model
- modify the model
- place orders
- connect to a broker
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

from system_config import (
    MODEL_FILE,
    MODEL_SHA256,
    FEATURE_FILE,
    FEATURE_COUNT,
    REAL_TRADING_ENABLED,
    BROKER_CONNECTION_ENABLED,
    MODEL_TRAINING_ENABLED,
    MODE,
)


ROOT = Path(__file__).resolve().parent.parent

TRAINING_FILE = ROOT / "data" / "market_features_v14.csv"


def check_safety():

    assert MODE == "PAPER_ONLY"
    assert REAL_TRADING_ENABLED is False
    assert BROKER_CONNECTION_ENABLED is False
    assert MODEL_TRAINING_ENABLED is False

    print("SAFETY: PAPER ONLY [PASS]")


def check_model_hash():

    sha = hashlib.sha256()

    with MODEL_FILE.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    actual = sha.hexdigest()

    if actual != MODEL_SHA256:
        raise RuntimeError(
            "FROZEN MODEL HASH MISMATCH\n"
            f"Expected: {MODEL_SHA256}\n"
            f"Actual:   {actual}"
        )

    print("MODEL HASH: [PASS]")


def check_feature_schema():

    training = pd.read_csv(TRAINING_FILE)
    live = pd.read_csv(FEATURE_FILE)

    training_features = [
        c for c in training.columns
        if c not in {
            "Date",
            "target",
            "future_5d_return",
        }
    ]

    live_features = [
        c for c in live.columns
        if c != "Date"
    ]

    if len(training_features) != FEATURE_COUNT:
        raise RuntimeError(
            "Training feature count mismatch."
        )

    if len(live_features) != FEATURE_COUNT:
        raise RuntimeError(
            "Live feature count mismatch."
        )

    if training_features != live_features:
        missing = [
            c for c in training_features
            if c not in live_features
        ]

        extra = [
            c for c in live_features
            if c not in training_features
        ]

        raise RuntimeError(
            "FEATURE SCHEMA MISMATCH\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    print(
        f"FEATURE SCHEMA: {FEATURE_COUNT}/{FEATURE_COUNT} [PASS]"
    )


def check_live_data():

    live = pd.read_csv(FEATURE_FILE)

    if live.empty:
        raise RuntimeError(
            "Live feature dataset is empty."
        )

    if live["Date"].isna().any():
        raise RuntimeError(
            "Live dataset contains invalid dates."
        )

    features = live.drop(columns=["Date"])

    if features.isna().any().any():
        raise RuntimeError(
            "Live dataset contains NaN values."
        )

    if not features.apply(
        lambda col: pd.api.types.is_numeric_dtype(col)
    ).all():
        raise RuntimeError(
            "Live dataset contains non-numeric features."
        )

    print(
        f"LIVE DATA: {len(live)} rows [PASS]"
    )

    print(
        f"LATEST DATE: {live['Date'].max()} [PASS]"
    )


def main():

    print()
    print("=" * 60)
    print("V28 PRODUCTION PREFLIGHT")
    print("=" * 60)

    check_safety()
    check_model_hash()
    check_feature_schema()
    check_live_data()

    print()
    print("=" * 60)
    print("PREFLIGHT: PASS")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 60)
        print("PREFLIGHT: FAIL")
        print("=" * 60)
        print(exc)
        sys.exit(1)
