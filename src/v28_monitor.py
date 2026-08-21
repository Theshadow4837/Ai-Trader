"""
V28 data and prediction monitor.

Paper-only.
Does not train, modify, or deploy the model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

LIVE_FILE = ROOT / "data" / "live_features_v24.csv"
PREDICTIONS_FILE = ROOT / "data" / "v28_paper_log.csv"
MODEL_FILE = ROOT / "models" / "v28" / "v28_seed_202_FROZEN.zip"

EXPECTED_FEATURE_COUNT = 85

EXPECTED_HASH = (
    "eb9c2dcd1e9cc1ac1ae85f1a4e759aae4aad57b10c9a0ce7a1320dfefd2245ad"
)


def check_features():
    df = pd.read_csv(LIVE_FILE)

    feature_columns = [
        c for c in df.columns
        if c != "Date"
    ]

    assert len(feature_columns) == EXPECTED_FEATURE_COUNT, (
        f"Expected {EXPECTED_FEATURE_COUNT} features, "
        f"got {len(feature_columns)}"
    )

    values = df[feature_columns]

    assert np.isfinite(values.to_numpy()).all(), (
        "NaN or infinite feature values detected"
    )

    return {
        "rows": len(df),
        "features": len(feature_columns),
        "latest_date": str(df["Date"].iloc[-1]),
    }


def check_predictions():
    df = pd.read_csv(PREDICTIONS_FILE)

    if len(df) == 0:
        return {
            "rows": 0,
            "actions": {},
        }

    # Find the action/prediction column without assuming
    # a particular column name.
    possible = [
        "action",
        "prediction",
        "predicted_action",
    ]

    action_column = next(
        (c for c in possible if c in df.columns),
        None,
    )

    if action_column is None:
        raise AssertionError(
            "Could not find prediction/action column"
        )

    actions = (
        df[action_column]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "rows": len(df),
        "actions": {
            str(k): int(v)
            for k, v in actions.items()
        },
    }


def check_model_hash():
    import hashlib

    digest = hashlib.sha256()

    with MODEL_FILE.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    actual = digest.hexdigest()

    assert actual == EXPECTED_HASH, (
        "FROZEN MODEL HASH MISMATCH"
    )

    return actual


def main():
    print("=" * 60)
    print("V28 MODEL / DATA MONITOR")
    print("=" * 60)

    features = check_features()

    print()
    print("FEATURE MONITOR")
    print(f"Rows:             {features['rows']}")
    print(f"Features:         {features['features']}")
    print(f"Latest date:      {features['latest_date']}")
    print("Finite values:    PASS")

    predictions = check_predictions()

    print()
    print("PREDICTION MONITOR")
    print(f"Prediction rows:  {predictions['rows']}")
    print(f"Actions:          {predictions['actions']}")

    model_hash = check_model_hash()

    print()
    print("MODEL MONITOR")
    print("Frozen hash:      PASS")
    print(f"SHA-256:          {model_hash}")

    print()
    print("PAPER ONLY")
    print("Training:         DISABLED")
    print("Real orders:      DISABLED")
    print("Broker:           DISABLED")

    print()
    print("=" * 60)
    print("V28 MONITOR: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
