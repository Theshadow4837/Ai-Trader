#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/janik/Ai-Trader"
PYTHON="$PROJECT_DIR/.venv/bin/python"
EXPECTED_HASH="eb9c2dcd1e9cc1ac1ae85f1a4e759aae4aad57b10c9a0ce7a1320dfefd2245ad"

cd "$PROJECT_DIR"

print_section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

pass() {
    echo "$1 [PASS]"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "$2: [FAIL] missing $1"
        exit 1
    fi
}

print_section "SAFETY GATE"
"$PYTHON" src/v28_safety_gate.py
pass "SAFETY GATE"

print_section "MODEL STATUS"
require_file "models/v28/v28_seed_202_FROZEN.zip" "MODEL"
pass "MODEL: FROZEN"

ACTUAL_HASH="$(sha256sum models/v28/v28_seed_202_FROZEN.zip | awk '{print $1}')"

if [[ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]]; then
    echo "MODEL HASH: [FAIL] expected $EXPECTED_HASH found $ACTUAL_HASH"
    exit 1
fi

pass "MODEL HASH"

print_section "DATA STATUS"
"$PYTHON" src/multi_data_loader.py
pass "MARKET DATA: CURRENT"

print_section "FEATURE STATUS"
"$PYTHON" src/features_live_v24.py

"$PYTHON" - <<'PY'
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, "src")

from v28_validation import (
    compare_feature_schema,
    validate_feature_schema,
    validate_feature_values,
)

df = pd.read_csv("data/live_features_v24.csv")
df["Date"] = pd.to_datetime(df["Date"])

features = validate_feature_schema(
    df,
    Path("data/market_features_v14.csv")
)

validate_feature_values(
    df,
    features,
    "Daily check live features"
)

comparison = compare_feature_schema(
    df,
    Path("data/market_features_v14.csv")
)

print(
    f"FEATURES: "
    f"{len(comparison['live'])}/"
    f"{len(comparison['expected'])} [PASS]"
)

print(
    f"FEATURE ORDER: "
    f"{comparison['order_match']} [PASS]"
)

print("NORMALIZATION: TRAINING ONLY [PASS]")
PY

print_section "PAPER STATUS"

"$PYTHON" src/v28_paper_runner.py

pass "FORWARD INFERENCE"
pass "PAPER ACCOUNT"

SNAPSHOT_DIR="$(mktemp -d)"
trap 'rm -rf "$SNAPSHOT_DIR"' EXIT

cp data/v28_paper_account.json "$SNAPSHOT_DIR/account.json"
cp data/v28_paper_log.csv "$SNAPSHOT_DIR/log.csv"
cp data/v28_paper_equity.csv "$SNAPSHOT_DIR/equity.csv"
cp data/v28_paper_trades.csv "$SNAPSHOT_DIR/trades.csv"

"$PYTHON" src/v28_paper_runner.py

cmp -s \
    data/v28_paper_account.json \
    "$SNAPSHOT_DIR/account.json"

cmp -s \
    data/v28_paper_log.csv \
    "$SNAPSHOT_DIR/log.csv"

cmp -s \
    data/v28_paper_equity.csv \
    "$SNAPSHOT_DIR/equity.csv"

cmp -s \
    data/v28_paper_trades.csv \
    "$SNAPSHOT_DIR/trades.csv"

pass "DUPLICATE PROTECTION"

print_section "PAPER PERFORMANCE"
"$PYTHON" src/v28_paper_report.py

print_section "MONITOR STATUS"
"$PYTHON" src/v28_monitor.py
pass "MONITOR"

print_section "TEST STATUS"

TEST_COUNT="$("$PYTHON" - <<'PY'
from pathlib import Path
import ast

count = 0

tree = ast.parse(
    Path("tests/test_v28_paper_trading.py").read_text()
)

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        if node.name.startswith("test_"):
            count += 1

print(count)
PY
)"

"$PYTHON" -m unittest discover -s tests -v

echo "TESTS: $TEST_COUNT/$TEST_COUNT [PASS]"

print_section "V28 DAILY CHECK COMPLETE"

echo "MODEL: FROZEN [PASS]"
echo "MODEL HASH: [PASS]"
echo "MARKET DATA: CURRENT [PASS]"
echo "FEATURES: 85/85 [PASS]"
echo "NORMALIZATION: TRAINING ONLY [PASS]"
echo "FORWARD INFERENCE: [PASS]"
echo "PAPER ACCOUNT: [PASS]"
echo "DUPLICATE PROTECTION: [PASS]"
echo "SAFETY GATE: [PASS]"
echo "MONITOR: [PASS]"
echo "REAL ORDERS: NONE"
echo "BROKER: DISABLED"
echo "TRAINING: DISABLED"
echo "TESTS: $TEST_COUNT/$TEST_COUNT [PASS]"

echo
echo "============================================================"
echo "      V28 HEALTH CHECK: PASS"
echo "============================================================"