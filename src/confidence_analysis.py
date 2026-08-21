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

    validation = data[
        data["Date"].dt.year.between(
            2022,
            2024
        )
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

    results = validation[
        ["Date", "target"]
    ].copy()

    results["prob_up"] = probabilities

    # Confidence = distance from 50%
    results["confidence"] = (
        abs(results["prob_up"] - 0.5)
    )

    bins = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.50
    ]

    labels = [
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80%+"
    ]

    results["confidence_group"] = pd.cut(
        results["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    print()
    print("================================")
    print("      CONFIDENCE ANALYSIS")
    print("================================")

    grouped = (
        results
        .groupby(
            "confidence_group",
            observed=True
        )
        .agg(
            predictions=("target", "count"),
            actual_up_rate=("target", "mean"),
            average_probability=("prob_up", "mean")
        )
    )

    grouped["actual_up_rate"] *= 100

    print(
        grouped.to_string()
    )


if __name__ == "__main__":
    main()