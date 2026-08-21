"""
============================================================
V28 CENTRAL CONFIGURATION
============================================================

Phase 15.1

Single source of truth for the V28 paper-trading system.

IMPORTANT:
- Paper trading only.
- No broker connection.
- No real orders.
- Frozen model must never be modified.
"""

from pathlib import Path


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v28"
    / "v28_seed_202_FROZEN.zip"
)

MODEL_SHA256 = (
    "eb9c2dcd1e9cc1ac1ae85f1a4e759aae4aad57b10c9a0ce7a1320dfefd2245ad"
)

MODEL_VERSION = "V28"
MODEL_STATUS = "FROZEN"


# ============================================================
# MARKET DATA
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

SPY_FILE = DATA_DIR / "SPY.csv"
QQQ_FILE = DATA_DIR / "QQQ.csv"
IWM_FILE = DATA_DIR / "IWM.csv"
DIA_FILE = DATA_DIR / "DIA.csv"
VIX_FILE = DATA_DIR / "VIX.csv"

LIVE_FEATURES_FILE = (
    DATA_DIR / "live_features_v24.csv"
)


# ============================================================
# PAPER TRADING
# ============================================================

INITIAL_CAPITAL = 10_000.0

TRANSACTION_COST = 0.0005

PAPER_ONLY = True
REAL_ORDERS_ENABLED = False
TRAINING_ENABLED = False


# ============================================================
# FEATURE CONTRACT
# ============================================================

EXPECTED_FEATURE_COUNT = 85

EXPECTED_OBSERVATION_SHAPE = (
    EXPECTED_FEATURE_COUNT,
)

NORMALIZATION_CLIP_MIN = -10.0
NORMALIZATION_CLIP_MAX = 10.0


# ============================================================
# TRAINING PERIOD
# ============================================================

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-29"


# ============================================================
# OUTPUT FILES
# ============================================================

PAPER_LOG_FILE = (
    DATA_DIR / "v28_paper_log.csv"
)

PAPER_ACCOUNT_FILE = (
    DATA_DIR / "v28_paper_account.json"
)

PAPER_EQUITY_FILE = (
    DATA_DIR / "v28_paper_equity.csv"
)

PAPER_TRADES_FILE = (
    DATA_DIR / "v28_paper_trades.csv"
)

PAPER_REPORT_FILE = (
    DATA_DIR / "v28_paper_report.txt"
)


# ============================================================
# REQUIRED MARKET SYMBOLS
# ============================================================

MARKET_SYMBOLS = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VIX",
)


# ============================================================
# SAFETY
# ============================================================

assert PAPER_ONLY is True
assert REAL_ORDERS_ENABLED is False
assert TRAINING_ENABLED is False

assert INITIAL_CAPITAL > 0
assert TRANSACTION_COST >= 0

assert EXPECTED_FEATURE_COUNT == 85
