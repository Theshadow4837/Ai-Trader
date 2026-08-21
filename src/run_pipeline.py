from v28_run_log import log_run
"""
AI-Trader production pipeline entry point.

SAFE MODE:
- Runs production preflight first
- Updates market data
- Generates live features
- Runs frozen V28 paper inference
- Generates paper performance report
- NEVER places real orders
- NEVER trains the model
"""

import subprocess
import sys
from pathlib import Path

from system_config import (
    SYSTEM_NAME,
    SYSTEM_VERSION,
    MODE,
    REAL_TRADING_ENABLED,
    BROKER_CONNECTION_ENABLED,
    MODEL_TRAINING_ENABLED,
)


ROOT = Path(__file__).resolve().parent.parent


def run_step(name, command):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    result = subprocess.run(
        command,
        cwd=ROOT,
    )

    if result.returncode != 0:
        print()
        print(f"[FAIL] {name}")
        sys.exit(result.returncode)

    print()
    print(f"[PASS] {name}")


def safety_check():

    if MODE != "PAPER_ONLY":
        raise RuntimeError(
            "System is not configured for PAPER_ONLY mode."
        )

    if REAL_TRADING_ENABLED:
        raise RuntimeError(
            "Real trading is enabled."
        )

    if BROKER_CONNECTION_ENABLED:
        raise RuntimeError(
            "Broker connection is enabled."
        )

    if MODEL_TRAINING_ENABLED:
        raise RuntimeError(
            "Model training is enabled."
        )


def main():

    print()
    print("=" * 60)
    print(SYSTEM_NAME)
    print(SYSTEM_VERSION)
    print("=" * 60)

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    safety_check()

    print()
    print("REAL TRADING: DISABLED [PASS]")
    print("BROKER CONNECTION: DISABLED [PASS]")
    print("MODEL TRAINING: DISABLED [PASS]")
    print("MODE: PAPER_ONLY [PASS]")

    python = sys.executable

    # --------------------------------------------------------
    # PREFLIGHT
    # --------------------------------------------------------

    run_step(
        "STEP 1 — PRODUCTION PREFLIGHT",
        [
            python,
            "src/v28_preflight.py",
        ],
    )

    # --------------------------------------------------------
    # MARKET DATA
    # --------------------------------------------------------

    run_step(
        "STEP 2 — MARKET DATA UPDATE",
        [
            python,
            "src/multi_data_loader.py",
        ],
    )

    # --------------------------------------------------------
    # LIVE FEATURES
    # --------------------------------------------------------

    run_step(
        "STEP 3 — LIVE FEATURE GENERATION",
        [
            python,
            "src/features_live_v24.py",
        ],
    )

    # --------------------------------------------------------
    # PAPER INFERENCE
    # --------------------------------------------------------

    run_step(
        "STEP 4 — V28 PAPER INFERENCE",
        [
            python,
            "src/v28_paper_runner.py",
        ],
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    run_step(
        "STEP 5 — PAPER PERFORMANCE REPORT",
        [
            python,
            "src/v28_paper_report.py",
        ],
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(f"SYSTEM: {SYSTEM_NAME}")
    print(f"VERSION: {SYSTEM_VERSION}")
    print(f"MODE: {MODE}")

    print()
    print("REAL ORDERS: NONE")
    print("BROKER CONNECTION: NONE")
    print("MODEL TRAINING: NONE")

    # --------------------------------------------------------
    # RUN LOG
    # --------------------------------------------------------

    import pandas as pd

    live_file = ROOT / "data" / "live_features_v24.csv"
    account_file = ROOT / "data" / "v28_paper_account.json"

    live = pd.read_csv(live_file)

    latest_market_date = live["Date"].max()

    predictions = len(
        pd.read_csv(
            ROOT / "data" / "v28_paper_log.csv"
        )
    )

    equity = 0.0
    position = 0

    if account_file.exists():

        import json

        with account_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            account = json.load(f)

        equity = float(
            account.get(
                "equity",
                0.0
            )
        )

        position = int(
            account.get(
                "position",
                0
            )
        )

    log_run(
        status="PASS",
        latest_market_date=latest_market_date,
        predictions=predictions,
        equity=equity,
        position=position,
    )

    print()
    print("PIPELINE STATUS: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()