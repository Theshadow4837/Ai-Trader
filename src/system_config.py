"""
AI-TRADER SYSTEM CONFIGURATION

Production/paper-trading configuration for V28.

IMPORTANT:
- The V28 model is frozen.
- This file must never modify the model.
- No broker connection.
- No real orders.
"""

from pathlib import Path


# ============================================================
# SYSTEM IDENTITY
# ============================================================

SYSTEM_NAME = "AI-Trader"

SYSTEM_VERSION = "V28-PAPER-PRODUCTION-1.0"

MODEL_VERSION = "V28"

MODE = "PAPER_ONLY"


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

MODEL_DIR = PROJECT_ROOT / "models" / "v28"

LOG_DIR = DATA_DIR


# ============================================================
# FROZEN MODEL
# ============================================================

MODEL_FILE = (
    MODEL_DIR /
    "v28_seed_202_FROZEN.zip"
)

MODEL_SHA256 = (
    "eb9c2dcd1e9cc1ac1ae85f1a4e759aae4aad57b10c9a0ce7a1320dfefd2245ad"
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_COUNT = 85

FEATURE_FILE = (
    DATA_DIR /
    "live_features_v24.csv"
)


# ============================================================
# TRAINING PERIOD
# ============================================================

TRAIN_START = "2015-01-01"

TRAIN_END = "2023-12-29"


# ============================================================
# MARKET DATA
# ============================================================

MARKET_SYMBOLS = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VIX",
)


# ============================================================
# PAPER TRADING
# ============================================================

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005


# ============================================================
# SAFETY
# ============================================================

REAL_TRADING_ENABLED = False

BROKER_CONNECTION_ENABLED = False

MODEL_TRAINING_ENABLED = False


# ============================================================
# VALIDATION
# ============================================================

EXPECTED_ACTIONS = 2

EXPECTED_OBSERVATION_DIMENSIONS = (
    FEATURE_COUNT,
)


def print_config():
    """Print the active system configuration."""

    print()
    print("=" * 60)
    print("AI-TRADER SYSTEM CONFIGURATION")
    print("=" * 60)

    print()
    print(f"System:              {SYSTEM_NAME}")
    print(f"System version:      {SYSTEM_VERSION}")
    print(f"Model version:       {MODEL_VERSION}")
    print(f"Mode:                {MODE}")

    print()
    print(f"Model:               {MODEL_FILE.name}")
    print(f"Feature count:       {FEATURE_COUNT}")
    print(f"Training period:     {TRAIN_START} → {TRAIN_END}")

    print()
    print(
        "Market symbols:      "
        + ", ".join(MARKET_SYMBOLS)
    )

    print()
    print(
        f"Real trading:        "
        f"{'ENABLED' if REAL_TRADING_ENABLED else 'DISABLED'}"
    )

    print(
        f"Broker connection:   "
        f"{'ENABLED' if BROKER_CONNECTION_ENABLED else 'DISABLED'}"
    )

    print(
        f"Model training:      "
        f"{'ENABLED' if MODEL_TRAINING_ENABLED else 'DISABLED'}"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    print_config()
