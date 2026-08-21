import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


DATA_FILE = Path("data/SPY_features_v6.csv")

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",

    "distance_sma10",
    "distance_sma20",
    "distance_sma50",
    "distance_sma100",
    "distance_sma200",

    "ema20_vs_ema50",

    "momentum_10d",
    "momentum_30d",
    "momentum_90d",

    "rsi_14",

    "volatility_10d",
    "volatility_20d",
    "volatility_60d",

    "atr_percent",

    "daily_range",
    "close_position",

    "volume_change",
    "volume_ratio",
    "volume_momentum_20d",
]


def main():

    data = pd.read_csv(DATA_FILE)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .dropna(
            subset=FEATURES + ["target"]
        )
        .reset_index(drop=True)
    )

    train = data[
        data["Date"].dt.year <= 2021
    ]

    X = train[FEATURES]
    y = train["target"]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest...")

    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print()
    print("================================")
    print("       FEATURE IMPORTANCE")
    print("================================")

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:25s}"
            f" {row['importance']:.6f}"
        )


if __name__ == "__main__":
    main()