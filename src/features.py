import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/SPY.csv")
OUTPUT_FILE = Path("data/SPY_features.csv")


def create_features():

    data = pd.read_csv(INPUT_FILE)

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # =========================
    # RETURNS
    # =========================

    data["return_1d"] = data["Close"].pct_change()

    data["return_5d"] = data["Close"].pct_change(5)

    data["return_20d"] = data["Close"].pct_change(20)

    # =========================
    # MOVING AVERAGES
    # =========================

    data["sma_10"] = data["Close"].rolling(10).mean()

    data["sma_50"] = data["Close"].rolling(50).mean()

    # Relative distance from moving averages
    data["distance_sma10"] = (
        data["Close"] / data["sma_10"] - 1
    )

    data["distance_sma50"] = (
        data["Close"] / data["sma_50"] - 1
    )

    # =========================
    # VOLATILITY
    # =========================

    data["volatility_10d"] = (
        data["return_1d"].rolling(10).std()
    )

    data["volatility_20d"] = (
        data["return_1d"].rolling(20).std()
    )

    # =========================
    # VOLUME
    # =========================

    data["volume_change"] = (
        data["Volume"].pct_change()
    )

    data["volume_ratio"] = (
        data["Volume"]
        / data["Volume"].rolling(20).mean()
    )

    # =========================
    # PRICE RANGE
    # =========================

    data["daily_range"] = (
        (data["High"] - data["Low"])
        / data["Close"]
    )

    # =========================
    # TARGET
    # =========================

    data["target"] = (
        data["Close"].shift(-1) > data["Close"]
    ).astype(int)

    # Remove rows with missing values
    data = data.dropna().reset_index(drop=True)

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Created {len(data)} feature rows."
    )

    print(
        f"Saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    create_features()