import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "market_features_v9.csv"

TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VIX",
]


def load_market(name):

    file = DATA_DIR / f"{name}.csv"

    data = pd.read_csv(file)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ]
        .sort_values("Date")
        .drop_duplicates("Date")
    )

    return data


def add_market_features(data, prefix):

    data = data.copy()

    close = data["Close"]

    data[f"{prefix}_return_1d"] = (
        close.pct_change(1)
    )

    data[f"{prefix}_return_5d"] = (
        close.pct_change(5)
    )

    data[f"{prefix}_return_20d"] = (
        close.pct_change(20)
    )

    data[f"{prefix}_return_60d"] = (
        close.pct_change(60)
    )

    data[f"{prefix}_volatility_10d"] = (
        data[f"{prefix}_return_1d"]
        .rolling(10)
        .std()
    )

    data[f"{prefix}_volatility_20d"] = (
        data[f"{prefix}_return_1d"]
        .rolling(20)
        .std()
    )

    sma_20 = (
        close.rolling(20).mean()
    )

    sma_50 = (
        close.rolling(50).mean()
    )

    sma_200 = (
        close.rolling(200).mean()
    )

    data[f"{prefix}_distance_sma20"] = (
        close / sma_20 - 1
    )

    data[f"{prefix}_distance_sma50"] = (
        close / sma_50 - 1
    )

    data[f"{prefix}_distance_sma200"] = (
        close / sma_200 - 1
    )

    data[f"{prefix}_daily_range"] = (
        (data["High"] - data["Low"])
        / close
    )

    # Volume features are not useful for VIX,
    # but are useful for the ETFs.

    if prefix != "VIX":

        volume_sma20 = (
            data["Volume"]
            .rolling(20)
            .mean()
        )

        data[f"{prefix}_volume_ratio"] = (
            data["Volume"]
            / volume_sma20
        )

        data[f"{prefix}_volume_change"] = (
            data["Volume"].pct_change()
        )

    return data


def main():

    print("================================")
    print("      V9 FEATURE ENGINE")
    print("================================")

    combined = None

    for ticker in TICKERS:

        print(
            f"Loading {ticker}..."
        )

        data = load_market(ticker)

        data = add_market_features(
            data,
            ticker
        )

        # Keep only the Date and
        # generated feature columns.

        keep = ["Date"]

        for column in data.columns:

            if column.startswith(
                f"{ticker}_"
            ):
                keep.append(column)

        data = data[keep]

        if combined is None:

            combined = data

        else:

            combined = combined.merge(
                data,
                on="Date",
                how="inner"
            )

    # =========================
    # SPY TARGET
    # =========================

    spy = load_market("SPY")

    combined = combined.merge(
        spy[
            [
                "Date",
                "Close"
            ]
        ],
        on="Date",
        how="inner"
    )

    combined["target"] = (
        combined["Close"].shift(-1)
        > combined["Close"]
    ).astype(int)

    combined.drop(
        columns=["Close"],
        inplace=True
    )

    # Remove rows where indicators
    # don't have enough history.

    combined = combined.dropna(
        subset=["target"]
    )

    combined = combined.dropna()

    combined = (
        combined
        .sort_values("Date")
        .reset_index(drop=True)
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    feature_columns = [
        column
        for column in combined.columns
        if column not in [
            "Date",
            "target"
        ]
    ]

    print()
    print(
        f"Saved {len(combined)} rows"
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print()
    print(
        f"Date range: "
        f"{combined['Date'].min().date()} "
        f"→ "
        f"{combined['Date'].max().date()}"
    )

    print()
    print("Features:")

    for feature in feature_columns:

        print(
            f" - {feature}"
        )


if __name__ == "__main__":
    main()