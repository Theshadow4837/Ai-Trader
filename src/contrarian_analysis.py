import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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

    data["Date"] = pd.to_datetime(data["Date"])

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

    validation = data[
        data["Date"].dt.year.between(2022, 2024)
    ]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")

    model.fit(
        train[FEATURES],
        train["target"]
    )

    probabilities = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    normal_predictions = (
        probabilities >= 0.5
    ).astype(int)

    contrarian_predictions = (
        probabilities < 0.5
    ).astype(int)

    normal_accuracy = accuracy_score(
        validation["target"],
        normal_predictions
    )

    contrarian_accuracy = accuracy_score(
        validation["target"],
        contrarian_predictions
    )

    print()
    print("================================")
    print("       CONTRARIAN ANALYSIS")
    print("================================")

    print(
        f"\nNormal accuracy: "
        f"{normal_accuracy:.4f}"
    )

    print(
        f"Contrarian accuracy: "
        f"{contrarian_accuracy:.4f}"
    )

    print()

    if contrarian_accuracy > normal_accuracy:
        print(
            "Contrarian signal performs better."
        )
    else:
        print(
            "Normal signal performs better."
        )


if __name__ == "__main__":
    main()