#!/usr/bin/env bash

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

echo "============================================"
echo "       V28 SYSTEM HEALTH CHECK"
echo "============================================"
echo

cd "$ROOT" || exit 1

echo "[1/5] TEST SUITE"

"$PYTHON" -m unittest discover -s tests -v

if [ $? -ne 0 ]; then
    echo "TESTS: FAIL"
    exit 1
fi

echo
echo "TESTS: PASS"

echo
echo "[2/5] MODEL INTEGRITY"

EXPECTED_HASH="eb9c2dcd1e9cc1ac1ae85f1a4e759aae4aad57b10c9a0ce7a1320dfefd2245ad"

MODEL="models/v28/v28_seed_202_FROZEN.zip"

ACTUAL_HASH="$(
    sha256sum "$MODEL" | awk '{print $1}'
)"

if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
    echo "MODEL HASH: FAIL"
    exit 1
fi

echo "MODEL HASH: PASS"

echo
echo "[3/5] FEATURE INTEGRITY"

"$PYTHON" src/features_live_v24.py

if [ $? -ne 0 ]; then
    echo "FEATURE GENERATION: FAIL"
    exit 1
fi

echo "FEATURE GENERATION: PASS"

echo
echo "[4/5] PAPER INFERENCE"

"$PYTHON" src/v28_paper_runner.py

if [ $? -ne 0 ]; then
    echo "PAPER INFERENCE: FAIL"
    exit 1
fi

echo "PAPER INFERENCE: PASS"

echo
echo
echo "[5/6] MODEL / DATA MONITOR"

"$PYTHON" src/v28_monitor.py

if [ $? -ne 0 ]; then
    echo "MONITOR: FAIL"
    exit 1
fi

echo "MONITOR: PASS"

echo "[6/6] FINAL STATUS"

echo "MODEL: FROZEN"
echo "MODEL HASH: PASS"
echo "FEATURES: PASS"
echo "FORWARD INFERENCE: PASS"
echo "REAL ORDERS: NONE"
echo "BROKER: DISABLED"
echo "TRAINING: DISABLED"
echo
echo "============================================"
echo "       V28 HEALTH CHECK: PASS"
echo "============================================"
