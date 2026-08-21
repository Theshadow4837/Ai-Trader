import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/market_features_v9.csv")
SPY_FILE = Path("data/SPY.csv")
OUTPUT_FILE = Path("data/market_features_v13.csv")


def main():
    print("Loading V9 features...")

    features = pd.read_csv(INPUT_FILE)
    spy = pd.read_csv(SPY_FILE)

    features["Date"] = pd.to_datetime(features["Date"])
    spy["Date"] = pd.to_datetime(spy["Date"])

    spy = (
        spy
        .sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    # Return over the NEXT 5 trading sessions.
    spy["future_5d_return"] = (
        spy["Close"].shift(-5) / spy["Close"] - 1
    )

    target = spy[
        ["Date", "future_5d_return"]
    ]

    data = features.merge(
        target,
        on="Date",
        how="inner"
    )

    data = data.dropna(
        subset=["future_5d_return"]
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if "target" in data.columns:
        data = data.drop(columns=["target"])

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    feature_count = len([
        c for c in data.columns
        if c not in [
            "Date",
            "future_5d_return"
        ]
    ])

    print()
    print(f"Saved {len(data)} rows")
    print(f"Feature count: {feature_count}")

    print(
        f"Date range: "
        f"{data['Date'].min().date()} → "
        f"{data['Date'].max().date()}"
    )

    print()
    print("5-day target statistics:")
    print(data["future_5d_return"].describe())


if __name__ == "__main__":
    main()