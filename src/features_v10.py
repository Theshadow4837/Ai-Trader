import pandas as pd
from pathlib import Path


INPUT_FILE = Path(
    "data/market_features_v9.csv"
)

SPY_FILE = Path(
    "data/SPY.csv"
)

OUTPUT_FILE = Path(
    "data/market_features_v10.csv"
)


def main():

    print("Loading V9 features...")

    features = pd.read_csv(
        INPUT_FILE
    )

    spy = pd.read_csv(
        SPY_FILE
    )

    features["Date"] = pd.to_datetime(
        features["Date"]
    )

    spy["Date"] = pd.to_datetime(
        spy["Date"]
    )

    spy = (
        spy
        .sort_values("Date")
        .drop_duplicates("Date")
    )

    # Calculate tomorrow's return.
    spy["next_day_return"] = (
        spy["Close"].shift(-1)
        / spy["Close"]
        - 1
    )

    target = spy[
        [
            "Date",
            "next_day_return"
        ]
    ]

    data = features.merge(
        target,
        on="Date",
        how="inner"
    )

    # The final row has no future price.
    data = data.dropna(
        subset=["next_day_return"]
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Remove the old classification target.
    if "target" in data.columns:
        data = data.drop(
            columns=["target"]
        )

    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    feature_count = len([
        c for c in data.columns
        if c not in [
            "Date",
            "next_day_return"
        ]
    ])

    print()
    print(
        f"Saved {len(data)} rows"
    )

    print(
        f"Feature count: "
        f"{feature_count}"
    )

    print(
        f"Date range: "
        f"{data['Date'].min().date()} "
        f"→ "
        f"{data['Date'].max().date()}"
    )

    print()
    print("Target statistics:")

    print(
        data["next_day_return"].describe()
    )


if __name__ == "__main__":
    main()