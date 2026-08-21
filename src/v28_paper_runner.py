"""
============================================================
V28 FROZEN FORWARD PAPER RUNNER
============================================================

RESEARCH / PAPER TRADING ONLY.

Loads the FROZEN V28 model and makes forward-only predictions
on newly available market data.

IMPORTANT:
    - No training.
    - No model modification.
    - No real orders.
    - No future returns are used to select actions.
    - Automatically processes only NEW rows.
    - Duplicate dates are skipped.
    - Historical rows are not replayed once logged.

ACTIONS:
    0 = FLAT
    1 = LONG

The model remains:
    models/v28/v28_seed_202_FROZEN.zip
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from v28_paper_account import PaperPortfolio
from v28_validation import (
    EXPECTED_MODEL_SHA256,
    FORBIDDEN_FEATURE_WORDS,
    compare_feature_schema,
    find_features,
    sha256_file,
    validate_feature_schema as validate_v28_feature_schema,
    validate_feature_values,
    verify_model_hash,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "models/v28/v28_seed_202_FROZEN.zip"
)

DATA_FILE = Path(
    "data/live_features_v24.csv"
)

LOG_FILE = Path(
    "data/v28_paper_log.csv"
)

ACCOUNT_FILE = Path("data/v28_paper_account.json")
EQUITY_FILE = Path("data/v28_paper_equity.csv")
TRADES_FILE = Path("data/v28_paper_trades.csv")
SPY_FILE = Path("data/SPY.csv")
REFERENCE_FEATURE_FILE = Path("data/market_features_v14.csv")

TRAIN_START = pd.Timestamp(
    "2015-01-01"
)

TRAIN_END = pd.Timestamp(
    "2023-12-29"
)


# ============================================================
# FEATURE RULES
# ============================================================

def validate_feature_schema(df, features):
    """Lock the live observation schema to the V28 training feature order."""
    validated = validate_v28_feature_schema(df, REFERENCE_FEATURE_FILE)
    if features != validated:
        raise RuntimeError("Computed feature list changed during V28 schema validation.")
    comparison = compare_feature_schema(df, REFERENCE_FEATURE_FILE)
    print()
    print("V28 FEATURE INTEGRITY")
    print(f"85 training features: {len(comparison['expected'])}")
    print(f"85 live features:     {len(comparison['live'])}")
    print(f"MISSING FROM LIVE:    {comparison['missing'] or 'none'}")
    print(f"EXTRA IN LIVE:        {comparison['extra'] or 'none'}")
    print(f"ORDER MATCH:          {comparison['order_match']}")


def load_spy_prices():
    if not SPY_FILE.exists():
        raise FileNotFoundError(f"SPY price data not found: {SPY_FILE}")
    prices = pd.read_csv(SPY_FILE, usecols=["Date", "Close"])
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.normalize()
    prices = prices.drop_duplicates("Date", keep="last").set_index("Date")["Close"]
    if prices.isna().any():
        raise RuntimeError("SPY close prices contain missing values.")
    return prices


def update_paper_account(portfolio, decisions, prices):
    """Apply already-produced decisions to the local simulator in date order."""
    for row in decisions.sort_values("date").itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        if date not in prices.index:
            raise RuntimeError(f"No SPY close price available for paper date {date.date()}.")
        portfolio.process(date, int(row.action), float(prices.loc[date]), str(date.date()))


def validate_market_progress(df, existing_log):
    """Reject a dataset that has regressed behind persisted paper decisions."""
    if len(existing_log) == 0 or "date" not in existing_log.columns:
        return
    newest = pd.Timestamp(df["Date"].max()).normalize()
    processed = pd.Timestamp(existing_log["date"].max()).normalize()
    if newest < processed:
        raise RuntimeError(
            f"STALE MARKET DATA: newest dataset row {newest.date()} "
            f"is older than last processed row {processed.date()}."
        )


def bootstrap_account_if_needed(existing_log, prices):
    portfolio = PaperPortfolio(ACCOUNT_FILE, EQUITY_FILE, TRADES_FILE)
    if ACCOUNT_FILE.exists() or len(existing_log) == 0:
        return portfolio
    required = {"date", "action"}
    if not required.issubset(existing_log.columns):
        raise RuntimeError("Existing paper log cannot initialize account: date/action missing.")
    print("[V28] Initializing paper account from existing immutable decision log...")
    update_paper_account(portfolio, existing_log, prices)
    return portfolio


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    if "Date" not in df.columns:

        raise RuntimeError(
            "Dataset must contain a Date column."
        )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TRAINING NORMALIZATION
# ============================================================

def get_training_stats(
    df,
    features
):

    train = df[
        (df["Date"] >= TRAIN_START)
        &
        (df["Date"] <= TRAIN_END)
    ].copy()

    train = train.dropna(
        subset=features
    )

    if len(train) == 0:

        raise RuntimeError(
            "No complete training rows available."
        )

    X = (
        train[features]
        .astype(np.float32)
        .to_numpy()
    )

    mean = np.nanmean(
        X,
        axis=0
    )

    std = np.nanstd(
        X,
        axis=0
    )

    std[
        std < 1e-8
    ] = 1.0

    return mean, std


# ============================================================
# LOAD EXISTING LOG
# ============================================================

def load_existing_log():

    if not LOG_FILE.exists():

        return pd.DataFrame()

    try:

        log = pd.read_csv(
            LOG_FILE
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read existing paper log:\n"
            f"{exc}"
        )

    if len(log) == 0:

        return pd.DataFrame()

    if "date" in log.columns:

        log["date"] = pd.to_datetime(
            log["date"]
        )

    return log


# ============================================================
# FIND NEW ROWS
# ============================================================

def get_new_rows(
    df,
    features,
    existing_log
):

    # --------------------------------------------------------
    # Only rows with complete model features.
    # --------------------------------------------------------

    paper = df.copy().reset_index(drop=True)

    if len(paper) == 0:

        raise RuntimeError(
            "No rows with complete features exist."
        )

    # --------------------------------------------------------
    # If we've already logged dates, remove them.
    # --------------------------------------------------------

    if (
        len(existing_log) > 0
        and "date" in existing_log.columns
    ):

        logged_dates = set(
            existing_log[
                "date"
            ]
            .dt.normalize()
            .tolist()
        )

        paper = paper[
            ~paper["Date"]
            .dt.normalize()
            .isin(logged_dates)
        ].copy()

    # --------------------------------------------------------
    # Sort newest last.
    # --------------------------------------------------------

    paper = (
        paper
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(paper) and paper[features].isna().any().any():
        validate_feature_values(paper, features, "New model observation")

    return paper


# ============================================================
# NORMALIZE
# ============================================================

def normalize_features(
    paper,
    features,
    mean,
    std
):

    X = (
        paper[features]
        .astype(np.float32)
        .to_numpy()
    )

    X = (
        (X - mean)
        / std
    )

    if not np.isfinite(X).all():
        raise RuntimeError("New model observation contains non-finite feature values.")

    X = np.clip(
        X,
        -10.0,
        10.0
    ).astype(np.float32)

    return X


# ============================================================
# RUN FORWARD INFERENCE
# ============================================================

def run_forward(
    model,
    paper,
    X,
    existing_log
):

    if len(paper) == 0:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Determine previous position.
    #
    # This is only for logging whether the new prediction
    # differs from the previous prediction.
    # --------------------------------------------------------

    if (
        len(existing_log) > 0
        and "position" in existing_log.columns
    ):

        previous_position = int(
            existing_log[
                "position"
            ].iloc[-1]
        )

        trade_count = int(
            existing_log[
                "trade_count"
            ].iloc[-1]
        )

    else:

        previous_position = 0
        trade_count = 0

    rows = []

    for i in range(
        len(paper)
    ):

        observation = X[i]

        # ----------------------------------------------------
        # Deterministic inference.
        # ----------------------------------------------------

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        action = int(
            np.asarray(
                action
            ).item()
        )

        # ----------------------------------------------------
        # V28 ACTIONS
        #
        # 0 = FLAT
        # 1 = LONG
        # ----------------------------------------------------

        if action == 1:

            new_position = 1

        elif action == 0:

            new_position = 0

        else:

            raise RuntimeError(
                f"Unexpected V28 action: {action}"
            )

        changed = (
            new_position
            != previous_position
        )

        if changed:

            trade_count += 1

        date = paper[
            "Date"
        ].iloc[i]

        rows.append({

            "date":
                date,

            "action":
                action,

            "position":
                new_position,

            "position_changed":
                changed,

            "trade_count":
                trade_count,

            "model":
                MODEL_PATH.name,

            "model_sha256":
                sha256_file(
                    MODEL_PATH
                ),

        })

        print(
            f"{date.date()} | "
            f"action={action} | "
            f"position={new_position} | "
            f"changed={changed}"
        )

        previous_position = (
            new_position
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE LOG
# ============================================================

def save_log(
    existing_log,
    new_log
):

    if len(new_log) == 0:

        print()
        print(
            "[V28] No new market rows "
            "to process."
        )

        return existing_log

    if len(existing_log) > 0:

        combined = pd.concat(
            [
                existing_log,
                new_log
            ],
            ignore_index=True
        )

    else:

        combined = new_log.copy()

    combined["date"] = pd.to_datetime(
        combined["date"]
    )

    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        LOG_FILE,
        index=False
    )

    return combined


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    log
):

    print()
    print("=" * 60)
    print("V28 FORWARD-ONLY PAPER SUMMARY")
    print("=" * 60)

    if len(log) == 0:

        print()
        print(
            "No predictions have been logged yet."
        )

        return

    first_date = pd.to_datetime(
        log["date"]
    ).min()

    last_date = pd.to_datetime(
        log["date"]
    ).max()

    trades = int(
        log[
            "position_changed"
        ].sum()
    )

    long_days = int(
        (
            log["position"] == 1
        ).sum()
    )

    time_in_market = (
        long_days
        / len(log)
    )

    current_position = int(
        log[
            "position"
        ].iloc[-1]
    )

    current_action = int(
        log[
            "action"
        ].iloc[-1]
    )

    print()
    print(
        f"Paper start:        "
        f"{first_date.date()}"
    )

    print(
        f"Paper latest:       "
        f"{last_date.date()}"
    )

    print(
        f"Predictions logged: "
        f"{len(log)}"
    )

    print(
        f"Trades:             "
        f"{trades}"
    )

    print(
        f"Time in market:     "
        f"{time_in_market * 100:.2f}%"
    )

    print(
        f"Latest action:      "
        f"{current_action}"
    )

    print(
        f"Latest position:    "
        f"{current_position}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "    No future return was used to select actions."
    )

    print(
        "    No real orders were placed."
    )

    print(
        "    The model was not trained."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V28 FROZEN AUTOMATIC PAPER RUNNER")
    print("=" * 60)
    print("PAPER TRADING ONLY")
    print("NO REAL ORDERS")

    # ========================================================
    # MODEL CHECK
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Frozen model not found:\n"
            f"{MODEL_PATH}"
        )

    model_hash = verify_model_hash(MODEL_PATH)

    print()
    print(
        "[V28] Loading FROZEN model..."
    )

    print(
        f"[V28] Model: "
        f"{MODEL_PATH}"
    )

    print(
        f"[V28] SHA-256: "
        f"{model_hash}"
    )

    if model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Frozen V28 model SHA-256 does not match the verified hash.")

    # --------------------------------------------------------
    # Load on CPU.
    # --------------------------------------------------------

    model = PPO.load(
        str(MODEL_PATH),
        device="cpu",
    )

    print(
        f"[V28] Action space: "
        f"{model.action_space}"
    )

    print(
        f"[V28] Observation space: "
        f"{model.observation_space}"
    )

    if model.action_space.n != 2:

        raise RuntimeError(
            "Frozen V28 model must use Discrete(2)."
        )

    # ========================================================
    # DATA
    # ========================================================

    df = load_data()

    features = find_features(
        df
    )

    validate_feature_schema(df, features)

    print()
    print(
        f"[V28] Total dataset rows: "
        f"{len(df)}"
    )

    print(
        f"[V28] Dataset range: "
        f"{df['Date'].iloc[0].date()} "
        f"→ "
        f"{df['Date'].iloc[-1].date()}"
    )

    print(
        f"[V28] Features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Model/feature compatibility.
    # --------------------------------------------------------

    expected_shape = (
        len(features),
    )

    if (
        model.observation_space.shape
        != expected_shape
    ):

        raise RuntimeError(
            "\nFeature/model dimension mismatch.\n"
            f"Model expects: "
            f"{model.observation_space.shape}\n"
            f"Dataset provides: "
            f"{expected_shape}"
        )

    print(
        "[V28] Feature dimensions: PASS"
    )

    # ========================================================
    # TRAINING NORMALIZATION
    # ========================================================

    mean, std = (
        get_training_stats(
            df,
            features
        )
    )

    print(
        "[V28] Training-only normalization: PASS"
    )

    # ========================================================
    # EXISTING LOG
    # ========================================================

    existing_log = (
        load_existing_log()
    )

    validate_market_progress(df, existing_log)

    prices = load_spy_prices()
    portfolio = bootstrap_account_if_needed(existing_log, prices)

    if len(existing_log) > 0:

        print()
        print(
            f"[V28] Existing predictions: "
            f"{len(existing_log)}"
        )

        print(
            f"[V28] Last logged date: "
            f"{pd.to_datetime(existing_log['date']).max().date()}"
        )

    else:

        print()
        print(
            "[V28] No previous paper predictions."
        )

    # ========================================================
    # NEW DATA
    # ========================================================

    paper = get_new_rows(
        df,
        features,
        existing_log
    )

    if len(paper) == 0:

        print()
        print("=" * 60)
        print("NO NEW MARKET DATA")
        print("=" * 60)

        print()
        print(
            f"Newest dataset row: "
            f"{df['Date'].iloc[-1].date()}"
        )

        if len(existing_log) > 0:

            print(
                f"Last processed row: "
                f"{pd.to_datetime(existing_log['date']).max().date()}"
            )

        print()
        print(
            "Nothing to predict yet."
        )

        print(
            "Update the market dataset and run this file again."
        )

        print()
        print(
            "MODEL STATUS:"
        )

        print(
            "    FROZEN"
        )

        print(
            "TRAINING:"
        )

        print(
            "    NONE"
        )

        return

    print()
    print(
        "[V28] NEW DATA FOUND"
    )

    print(
        f"[V28] New rows: "
        f"{len(paper)}"
    )

    print(
        f"[V28] New period: "
        f"{paper['Date'].iloc[0].date()} "
        f"→ "
        f"{paper['Date'].iloc[-1].date()}"
    )

    # ========================================================
    # NORMALIZE NEW DATA
    # ========================================================

    X = normalize_features(
        paper,
        features,
        mean,
        std
    )

    # ========================================================
    # FORWARD INFERENCE
    # ========================================================

    print()
    print(
        "[V28] Running forward-only inference..."
    )

    new_log = run_forward(
        model,
        paper,
        X,
        existing_log
    )

    # ========================================================
    # SAVE
    # ========================================================

    log = save_log(
        existing_log,
        new_log
    )

    # Persist the account only after its matching prediction records are saved.
    update_paper_account(portfolio, new_log, prices)

    print()
    print(
        "[V28] Decision log saved:"
    )

    print(
        f"       {LOG_FILE}"
    )

    print(f"[V28] Paper account saved: {ACCOUNT_FILE}")
    print(f"[V28] Daily equity saved:  {EQUITY_FILE}")
    print(f"[V28] Trade ledger saved:  {TRADES_FILE}")

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        log
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 60)
    print("V28 PAPER RUN COMPLETE")
    print("=" * 60)

    print()
    print(
        "MODEL STATUS:"
    )

    print(
        "    FROZEN"
    )

    print(
        "TRAINING:"
    )

    print(
        "    NONE"
    )

    print(
        "REAL ORDERS:"
    )

    print(
        "    NONE"
    )

    print(
        "FORWARD-ONLY INFERENCE:"
    )

    print(
        "    PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
