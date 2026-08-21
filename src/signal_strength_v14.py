import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

DATA_FILE = Path("data/market_features_v14.csv")


def main():

    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .dropna()
        .reset_index(drop=True)
    )

    FEATURES = [
        c for c in data.columns
        if c not in ["Date", "future_5d_return"]
    ]

    predictions = []

    # True walk-forward predictions
    for test_year in range(2020, 2027):

        train = data[data["Date"].dt.year < test_year]
        test = data[data["Date"].dt.year == test_year].copy()

        if len(test) == 0:
            continue

        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            loss="huber",
            random_state=42
        )

        model.fit(
            train[FEATURES],
            train["future_5d_return"]
        )

        test["prediction"] = model.predict(
            test[FEATURES]
        )

        predictions.append(test)

    results = pd.concat(predictions, ignore_index=True)

    # Rank predictions into percentile groups
    results["percentile"] = (
        results["prediction"]
        .rank(pct=True)
    )

    bins = [
        0.0,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        1.0
    ]

    labels = [
        "Bottom 10%",
        "10-25%",
        "25-50%",
        "50-75%",
        "75-90%",
        "Top 10%"
    ]

    results["signal_group"] = pd.cut(
        results["percentile"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    analysis = results.groupby(
        "signal_group",
        observed=False
    ).agg(
        predictions=("prediction", "count"),
        average_prediction=("prediction", "mean"),
        average_actual=("future_5d_return", "mean"),
        median_actual=("future_5d_return", "median"),
        actual_std=("future_5d_return", "std")
    )

    print()
    print("================================")
    print("   V13 SIGNAL STRENGTH ANALYSIS")
    print("================================")

    print(
        analysis.to_string(
            formatters={
                "average_prediction":
                    "{:.4%}".format,
                "average_actual":
                    "{:.4%}".format,
                "median_actual":
                    "{:.4%}".format,
                "actual_std":
                    "{:.4%}".format
            }
        )
    )

    # Top vs bottom
    top = results[
        results["percentile"] >= 0.90
    ]

    bottom = results[
        results["percentile"] <= 0.10
    ]

    print()
    print("================================")
    print("       TOP vs BOTTOM")
    print("================================")

    print(
        f"Top 10% average actual: "
        f"{top['future_5d_return'].mean():.4%}"
    )

    print(
        f"Bottom 10% average actual: "
        f"{bottom['future_5d_return'].mean():.4%}"
    )

    print(
        f"Difference: "
        f"{top['future_5d_return'].mean() - bottom['future_5d_return'].mean():.4%}"
    )

    # Correlation across all walk-forward predictions
    correlation = results[
        ["prediction", "future_5d_return"]
    ].corr().iloc[0, 1]

    print()
    print(
        f"Prediction/actual correlation: "
        f"{correlation:.4f}"
    )


if __name__ == "__main__":
    main()