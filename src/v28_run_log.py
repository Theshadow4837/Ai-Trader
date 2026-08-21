"""
V28 pipeline run logger.

Records the health/status of each pipeline execution.
Paper trading only.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = ROOT / "data" / "system_logs"
LOG_FILE = LOG_DIR / "pipeline_runs.jsonl"


def log_run(
    status,
    latest_market_date,
    predictions,
    equity,
    position,
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "system": "AI-Trader",

        "system_version":
            "V28-PAPER-PRODUCTION-1.0",

        "model_version":
            "V28",

        "mode":
            "PAPER_ONLY",

        "status":
            status,

        "latest_market_date":
            str(latest_market_date),

        "predictions_logged":
            int(predictions),

        "simulated_equity":
            float(equity),

        "current_position":
            int(position),

        "real_orders":
            False,

        "model_training":
            False,

        "broker_connection":
            False,
    }

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(record)
            + "\n"
        )

    print(
        f"RUN LOG SAVED: {LOG_FILE}"
    )


if __name__ == "__main__":

    log_run(
        status="TEST",
        latest_market_date="2026-08-19",
        predictions=2725,
        equity=1596299.21,
        position=0,
    )
