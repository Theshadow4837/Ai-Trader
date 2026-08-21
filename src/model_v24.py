import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier


# ============================================================
# V24 PAPER TRADER
#              V23 → LIVE FEATURES
#
# PAPER / RESEARCH ONLY
# ============================================================

TRAIN_FILE = Path("data/market_features_v14.csv")
LIVE_FILE = Path("data/live_features_v24.csv")
TRADES_FILE = Path("data/trades_v24.csv")
ACCOUNT_FILE = Path("data/account_v24.csv")

INITIAL_CAPITAL = 10_000.0

CONFIDENCE_THRESHOLD = 0.50

PROFIT_TARGET = 0.0020

TRANSACTION_COST = 0.0005

CHECK_INTERVAL = 60


RAW_PRICE_FEATURES = {
    "QQQ_close",
    "IWM_close",
    "DIA_close",
    "VIX_level",
}


# ============================================================
# MODEL
# ============================================================

def create_model():

    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=15,
        subsample=0.8,
        random_state=42,
    )


def calculate_sample_weights(returns):

    returns = np.asarray(returns)

    magnitude = np.abs(returns)

    normalized = np.clip(
        magnitude / 0.02,
        0,
        2,
    )

    return 1.0 + 0.50 * normalized


# ============================================================
# TRAINING DATA
# ============================================================

def load_training_data():

    data = pd.read_csv(TRAIN_FILE)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    data["trade_label"] = (
        data["future_5d_return"]
        > PROFIT_TARGET
    ).astype(int)

    exclude = {
        "Date",
        "target",

        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return",

        "trade_label",
    }

    features = [
        c
        for c in data.columns
        if c not in exclude
        and c not in RAW_PRICE_FEATURES
    ]

    data = (
        data
        .dropna(
            subset=features
            + [
                "future_5d_return",
                "trade_label",
            ]
        )
        .reset_index(drop=True)
    )

    return data, features


# ============================================================
# TRAIN V23
# ============================================================

def train_v23():

    print()
    print("[V24] Training V23...")

    data, features = load_training_data()

    model = create_model()

    weights = calculate_sample_weights(
        data["future_5d_return"]
    )

    model.fit(
        data[features],
        data["trade_label"],
        sample_weight=weights,
    )

    print(
        f"[V24] Training samples: {len(data)}"
    )

    print(
        f"[V24] Features: {len(features)}"
    )

    print(
        f"[V24] Training through: "
        f"{data['Date'].max().date()}"
    )

    return model, features


# ============================================================
# ACCOUNT
# ============================================================

def load_account():

    if not ACCOUNT_FILE.exists():

        save_account(
            INITIAL_CAPITAL
        )

        return INITIAL_CAPITAL

    try:

        account = pd.read_csv(
            ACCOUNT_FILE
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):

        save_account(
            INITIAL_CAPITAL
        )

        return INITIAL_CAPITAL

    if (
        account.empty
        or "equity" not in account.columns
    ):

        save_account(
            INITIAL_CAPITAL
        )

        return INITIAL_CAPITAL

    return float(
        account.iloc[-1]["equity"]
    )


def save_account(equity):

    total_return = (
        equity
        / INITIAL_CAPITAL
        - 1.0
    )

    account = pd.DataFrame({
        "cash": [equity],
        "equity": [equity],
        "total_return": [total_return],
    })

    account.to_csv(
        ACCOUNT_FILE,
        index=False,
    )


# ============================================================
# TRADE LEDGER
# ============================================================

TRADE_COLUMNS = [
    "Date",
    "probability",
    "threshold",
    "signal",
    "status",
    "virtual_return",
    "equity",
]


def load_trades():

    if not TRADES_FILE.exists():

        return pd.DataFrame(
            columns=TRADE_COLUMNS
        )

    try:

        trades = pd.read_csv(
            TRADES_FILE
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):

        print(
            "[V24] Invalid trades_v24.csv."
        )

        print(
            "[V24] Recreating ledger."
        )

        return pd.DataFrame(
            columns=TRADE_COLUMNS
        )

    if "Date" not in trades.columns:

        print(
            "[V24] trades_v24.csv has no Date column."
        )

        print(
            "[V24] Recreating ledger."
        )

        return pd.DataFrame(
            columns=TRADE_COLUMNS
        )

    for column in TRADE_COLUMNS:

        if column not in trades.columns:

            trades[column] = np.nan

    return trades[
        TRADE_COLUMNS
    ]


def save_trade(
    date,
    probability,
    signal,
    status,
    virtual_return,
    equity,
):

    trades = load_trades()

    row = pd.DataFrame([{
        "Date": str(date),

        "probability":
            float(probability),

        "threshold":
            float(
                CONFIDENCE_THRESHOLD
            ),

        "signal":
            int(signal),

        "status":
            status,

        "virtual_return":
            virtual_return,

        "equity":
            float(equity),
    }])

    trades = pd.concat(
        [
            trades,
            row,
        ],
        ignore_index=True,
    )

    trades.to_csv(
        TRADES_FILE,
        index=False,
    )

    print(
        f"[V24] Saved trade → "
        f"{TRADES_FILE}"
    )


def already_processed(date):

    trades = load_trades()

    if trades.empty:
        return False

    if "Date" not in trades.columns:
        return False

    dates = pd.to_datetime(
        trades["Date"],
        errors="coerce",
    )

    target = pd.Timestamp(
        date
    )

    return bool(
        (dates == target).any()
    )


# ============================================================
# LIVE DATA
# ============================================================

def get_latest_live_row():

    if not LIVE_FILE.exists():
        return None

    live = pd.read_csv(
        LIVE_FILE
    )

    if live.empty:
        return None

    if "Date" not in live.columns:

        raise RuntimeError(
            "live_features_v24.csv "
            "does not contain Date."
        )

    live["Date"] = pd.to_datetime(
        live["Date"]
    )

    live = (
        live
        .sort_values("Date")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return live.iloc[-1]


# ============================================================
# PREDICTION
# ============================================================

def predict_latest(
    model,
    features,
):

    live = pd.read_csv(
        LIVE_FILE
    )

    if live.empty:
        return None

    if "Date" not in live.columns:

        raise RuntimeError(
            "LIVE FILE ERROR: "
            "Date column missing."
        )

    live["Date"] = pd.to_datetime(
        live["Date"]
    )

    live = (
        live
        .sort_values("Date")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    row = live.iloc[-1]

    missing = [
        feature
        for feature in features
        if feature not in live.columns
    ]

    if missing:

        raise RuntimeError(
            "LIVE DATA MISSING FEATURES:\n"
            + "\n".join(missing)
        )

    future_columns = [
        c
        for c in live.columns
        if c.startswith("future_")
    ]

    if future_columns:

        raise RuntimeError(
            "Future columns found in live data: "
            + str(future_columns)
        )

    values = []

    for feature in features:
        values.append(
            row[feature]
        )

    X = pd.DataFrame(
        [values],
        columns=features,
    )

    if X.isna().any().any():

        missing_features = list(
            X.columns[
                X.isna().iloc[0]
            ]
        )

        raise RuntimeError(
            "NaN live features: "
            + str(missing_features)
        )

    probability = float(
        model.predict_proba(
            X
        )[0, 1]
    )

    return (
        row["Date"],
        probability,
    )


# ============================================================
# PROCESS SIGNAL
# ============================================================

def process_signal(
    model,
    features,
):

    result = predict_latest(
        model,
        features,
    )

    if result is None:

        print(
            "[V24] No live data."
        )

        return

    date, probability = result

    date_string = (
        pd.Timestamp(date)
        .strftime("%Y-%m-%d")
    )

    print()
    print(
        "================================"
    )
    print(
        f"[V24] NEW MARKET DATA: "
        f"{date_string}"
    )
    print(
        "================================"
    )

    print(
        f"[V24] V23 probability: "
        f"{probability:.4f}"
    )

    signal = int(
        probability
        >= CONFIDENCE_THRESHOLD
    )

    if signal:

        print(
            "[V24] SIGNAL: PAPER BUY"
        )

    else:

        print(
            "[V24] SIGNAL: NO TRADE"
        )

    if already_processed(date):

        print(
            "[V24] This date has already "
            "been processed."
        )

        return

    equity = load_account()

    if signal:

        save_trade(
            date=date_string,
            probability=probability,
            signal=1,
            status="OPEN",
            virtual_return=np.nan,
            equity=equity,
        )

        print(
            "[V24] Paper trade opened."
        )

    else:

        save_trade(
            date=date_string,
            probability=probability,
            signal=0,
            status="NO_TRADE",
            virtual_return=0.0,
            equity=equity,
        )

        print(
            "[V24] No paper trade opened."
        )

    print(
        f"[V24] Virtual equity: "
        f"${equity:,.2f}"
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print()
    print(
        "############################################################"
    )
    print(
        "#                    V24 PAPER TRADER"
    )
    print(
        "#              V23 → LIVE FEATURES"
    )
    print(
        "############################################################"
    )

    print()
    print(
        "Paper trading only."
    )

    model, features = train_v23()

    print()
    print(
        "[V24] Starting live monitoring..."
    )

    last_seen = None

    while True:

        try:

            latest = get_latest_live_row()

            if latest is None:

                print(
                    "[V24] No live feature data."
                )

            else:

                latest_date = pd.Timestamp(
                    latest["Date"]
                )

                if (
                    last_seen is None
                    or latest_date > last_seen
                ):

                    print()
                    print(
                        "[V24] New feature row "
                        "detected: "
                        f"{latest_date.date()}"
                    )

                    process_signal(
                        model,
                        features,
                    )

                    last_seen = latest_date

                else:

                    print(
                        "[V24] No new feature data. "
                        f"Latest: "
                        f"{latest_date.date()}. "
                        f"Next check in "
                        f"{CHECK_INTERVAL}s."
                    )

        except Exception as error:

            print()
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            print(
                "[V24 ERROR]"
            )
            print(
                repr(error)
            )
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

        time.sleep(
            CHECK_INTERVAL
        )


if __name__ == "__main__":
    main()