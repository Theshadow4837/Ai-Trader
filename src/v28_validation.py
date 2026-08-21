"""Shared validation for the frozen V28 paper-trading pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MODEL_SHA256 = "eb9c2dcd1e9cc1ac1ae85f1a4e759aae4aad57b10c9a0ce7a1320dfefd2245ad"
EXPECTED_FEATURE_COUNT = 85

FORBIDDEN_FEATURE_WORDS = {
    "future",
    "target",
    "label",
    "reward",
}

EXCLUDED_COLUMNS = {
    "Date",
    "future_1d_return",
    "future_3d_return",
    "future_5d_return",
    "future_10d_return",
    "target",
    "trade_reward",
    "trade_label",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_model_hash(model_path: Path, expected_hash: str = EXPECTED_MODEL_SHA256) -> str:
    actual_hash = sha256_file(model_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "Frozen V28 model SHA-256 does not match the verified hash. "
            f"Expected {expected_hash}, found {actual_hash}."
        )
    return actual_hash


def find_features(df: pd.DataFrame) -> list[str]:
    features = []
    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        lower = column.lower()
        if lower in {"date", "datetime", "timestamp"}:
            continue
        if any(word in lower for word in FORBIDDEN_FEATURE_WORDS):
            continue
        features.append(column)
    return features


def expected_v28_features(reference_feature_file: Path) -> list[str]:
    if not reference_feature_file.exists():
        raise FileNotFoundError(f"V28 feature reference not found: {reference_feature_file}")
    reference = pd.read_csv(reference_feature_file, nrows=1)
    return find_features(reference)


def compare_feature_schema(df: pd.DataFrame, reference_feature_file: Path) -> dict:
    live_features = find_features(df)
    expected = expected_v28_features(reference_feature_file)
    missing = [name for name in expected if name not in live_features]
    extra = [name for name in live_features if name not in expected]
    return {
        "expected": expected,
        "live": live_features,
        "missing": missing,
        "extra": extra,
        "order_match": live_features == expected,
    }


def validate_feature_schema(df: pd.DataFrame, reference_feature_file: Path) -> list[str]:
    """Validate V28 observation columns and return the ordered feature list."""
    forbidden = [
        column for column in df.columns
        if any(word in column.lower() for word in FORBIDDEN_FEATURE_WORDS)
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden future/target columns present: {forbidden}")

    comparison = compare_feature_schema(df, reference_feature_file)
    features = comparison["live"]
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"V28 requires exactly {EXPECTED_FEATURE_COUNT} features; found {len(features)}."
        )
    if comparison["missing"] or comparison["extra"] or not comparison["order_match"]:
        raise RuntimeError(
            "Live feature order does not match V28 training schema. "
            f"Missing: {comparison['missing']}; extra: {comparison['extra']}; "
            f"order_match: {comparison['order_match']}"
        )
    return features


def validate_feature_values(df: pd.DataFrame, features: list[str], context: str) -> None:
    values = df[features].replace([np.inf, -np.inf], np.nan)
    if values.isna().any().any():
        bad_dates = (
            df.loc[values.isna().any(axis=1), "Date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
            if "Date" in df.columns
            else []
        )
        suffix = f": {bad_dates}" if bad_dates else "."
        raise RuntimeError(f"{context} contains NaN or infinite feature values{suffix}")
