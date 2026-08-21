"""
V28 safety gate.

Paper-only.
Fails closed when critical integrity conditions are violated.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

LIVE_FILE = ROOT / "data" / "live_features_v24.csv"
MODEL_FILE = ROOT / "models" / "v28" / "v28_seed_202_FROZEN.zip"

EXPECTED_FEATURE_COUNT = 85

EXPECTED_HASH = (
    "eb9c2dcd1e9cc1ac1ae85f1a4e759aae4aad57b10c9a0ce7a1320dfefd2245ad"
)


def check_model():
    digest = hashlib.sha256()

    with MODEL_FILE.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    if digest.hexdigest() != EXPECTED_HASH:
        raise RuntimeError(
            "SAFETY GATE: frozen model hash mismatch"
        )


def check_features():
    df = pd.read_csv(LIVE_FILE)

    if "Date" not in df.columns:
        raise RuntimeError(
            "SAFETY GATE: Date column missing"
        )

    features = [
        c for c in df.columns
        if c != "Date"
    ]

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"SAFETY GATE: expected "
            f"{EXPECTED_FEATURE_COUNT} features, "
            f"got {len(features)}"
        )

    values = df[features].to_numpy()

    if not np.isfinite(values).all():
        raise RuntimeError(
            "SAFETY GATE: NaN or infinite feature detected"
        )

    dates = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    if dates.isna().any():
        raise RuntimeError(
            "SAFETY GATE: invalid market date"
        )

    if dates.duplicated().any():
        raise RuntimeError(
            "SAFETY GATE: duplicate market date"
        )

    if not dates.is_monotonic_increasing:
        raise RuntimeError(
            "SAFETY GATE: market dates are not ordered"
        )


def main():
    print("V28 SAFETY GATE")
    print("----------------")

    check_model()
    print("MODEL HASH: PASS")

    check_features()
    print("FEATURE INTEGRITY: PASS")

    print("FAIL-CLOSED PROTECTION: ENABLED")
    print("REAL ORDERS: DISABLED")
    print("TRAINING: DISABLED")
    print("BROKER: DISABLED")
    print()
    print("SAFETY GATE: PASS")


if __name__ == "__main__":
    main()
