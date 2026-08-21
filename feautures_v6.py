import pandas as pd
import numpy as np

from pathlib import Path


INPUT_FILE = Path("data/SPY.csv")
OUTPUT_FILE = Path("data/SPY_features_v6.csv")


def add_features(data):

    data = data.copy()

    # =========================
    # RETURNS
    # =========================

    data["return_1d"] = (
        data["Close"].pct_change(1)
    )

    data["return_5d"] = (
        data["Close"].pct_change(5)
    )

    data["return_20d"] = (
        data["Close"].pct_change(20)
    )

    data["return_60d"] = (
        data["Close"].pct_change(60)
    )

    # =========================
    # MOVING AVERAGES
    # =========================

    data["sma_10"] = (
        data["Close"].rolling(10).mean()
    )

    data["sma_20"] = (
        data["Close"].rolling(20).mean()
    )

    data["sma_50"] = (
        data["Close"].rolling(50).mean()
    )

    data["sma_100"] = (
        data["Close"].rolling(100).mean()
    )

    data["sma_200"] = (
        data["Close"].rolling(200).mean()
    )

    data["ema_20"] = (
        data["Close"].ewm(
            span=20,
            adjust=False
        ).mean()
    )

    data["ema_50"] = (
        data["Close"].ewm(
            span=50,
            adjust=False
        ).mean()
    )

    # Distance from moving averages

    data["distance_sma10"] = (
        data["Close"] / data["sma_10"] - 1
    )

    data["distance_sma20"] = (
        data["Close"] / data["sma_20"] - 1
    )

    data["distance_sma50"] = (
        data["Close"] / data["sma_50"] - 1
    )

    data["distance_sma100"] = (
        data["Close"] / data["sma_100"] - 1
    )

    data["distance_sma200"] = (
        data["Close"] / data["sma_200"] - 1
    )

    data["ema20_vs_ema50"] = (
        data["ema_20"]
        / data["ema_50"]
        - 1
    )

    # =========================
    # MOMENTUM
    # =========================

    data["momentum_10d"] = (
        data["Close"]
        / data["Close"].shift(10)
        - 1
    )

    data["momentum_30d"] = (
        data["Close"]
        / data["Close"].shift(30)
        - 1
    )

    data["momentum_90d"] = (
        data["Close"]
        / data["Close"].shift(90)
        - 1
    )

    # =========================
    # RSI
    # =========================

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = (
        gain.rolling(14).mean()
    )

    avg_loss = (
        loss.rolling(14).mean()
    )

    rs = avg_gain / avg_loss

    data["rsi_14"] = (
        100 - (100 / (1 + rs))
    )

    # =========================
    # VOLATILITY
    # =========================

    data["volatility_10d"] = (
        data["return_1d"]
        .rolling(10)
        .std()
    )

    data["volatility_20d"] = (
        data["return_1d"]
        .rolling(20)
        .std()
    )

    data["volatility_60d"] = (
        data["return_1d"]
        .rolling(60)
        .std()
    )

    # =========================
    # TRUE RANGE / ATR
    # =========================

    previous_close = (
        data["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    data["atr_14"] = (
        true_range.rolling(14).mean()
    )

    data["atr_percent"] = (
        data["atr_14"]
        / data["Close"]
    )

    # =========================
    # PRICE RANGE
    # =========================

    data["daily_range"] = (
        (data["High"] - data["Low"])
        / data["Close"]
    )

    data["close_position"] = (
        (data["Close"] - data["Low"])
        / (
            data["High"]
            - data["Low"]
            + 1e-9
        )
    )

    # =========================
    # VOLUME
    # =========================

    data["volume_change"] = (
        data["Volume"].pct_change()
    )

    data["volume_sma20"] = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = (
        data["Volume"]
        / data["volume_sma20"]
    )

    data["volume_momentum_20d"] = (
        data["Volume"]
        / data["Volume"].shift(20)
        - 1
    )

    # =========================
    # TARGET
    # =========================

    # Predict whether the NEXT day's
    # closing price is higher.

    data["target"] = (
        data["Close"].shift(-1)
        > data["Close"]
    ).astype(int)

    return data


def main():

    print("Loading SPY data...")

    data = pd.read_csv(INPUT_FILE)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = data.sort_values(
        "Date"
    ).reset_index(drop=True)

    data = add_features(data)

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved {len(data)} rows to "
        f"{OUTPUT_FILE}"
    )

    print()
    print("Feature count:")

    feature_columns = [
        column
        for column in data.columns
        if column not in [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "target"
        ]
    ]

    print(
        len(feature_columns)
    )

    print()
    print("Features:")

    for feature in feature_columns:
        print(
            f" - {feature}"
        )


if __name__ == "__main__":
    main()