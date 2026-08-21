import pandas as pd

from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor


DATA_FILE = Path(
    "data/market_features_v10.csv"
)


def main():

    data = pd.read_csv(DATA_FILE)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .dropna()
        .reset_index(drop=True)
    )

    FEATURES = [
        column
        for column in data.columns
        if column not in [
            "Date",
            "next_day_return"
        ]
    ]

    # Use only information available before
    # each test year.

    all_results = []

    for test_year in range(2020, 2027):

        train = data[
            data["Date"].dt.year < test_year
        ]

        test = data[
            data["Date"].dt.year == test_year
        ].copy()

        if len(train) == 0 or len(test) == 0:
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
            train["next_day_return"]
        )

        test["prediction"] = model.predict(
            test[FEATURES]
        )

        test["prediction_pct"] = (
            test["prediction"] * 100
        )

        test["actual_pct"] = (
            test["next_day_return"] * 100
        )

        all_results.append(
            test[
                [
                    "Date",
                    "prediction",
                    "prediction_pct",
                    "actual_pct"
                ]
            ]
        )

    results = pd.concat(
        all_results,
        ignore_index=True
    )
    print()
    print("================================")
    print("   YEARLY SIGNAL ANALYSIS")
    print("================================")

    results["abs_prediction"] = (
        results["prediction"].abs()
    )

    bins = [
        0.0000,
        0.0010,
        0.0025,
        0.0050,
        0.0100,
        float("inf")
    ]

    labels = [
        "0.00-0.10%",
        "0.10-0.25%",
        "0.25-0.50%",
        "0.50-1.00%",
        "1.00%+"
    ]

    results["signal_strength"] = pd.cut(
        results["abs_prediction"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    for year in sorted(
        results["Date"].dt.year.unique()
    ):

        year_data = results[
            results["Date"].dt.year == year
        ]

        print()
        print(f"YEAR {year}")
        print("--------------------------------")

        grouped = (
            year_data
            .groupby(
                "signal_strength",
                observed=True
            )
            .agg(
                samples=(
                    "actual_pct",
                    "count"
                ),
                predicted=(
                    "prediction_pct",
                    "mean"
                ),
                actual=(
                    "actual_pct",
                    "mean"
                )
            )
        )

        print(
            grouped.to_string()
        )

        year_data = year_data.copy()

        year_data["correct"] = (
            (
                year_data["prediction"] >= 0
            )
            ==
            (
                year_data["actual_pct"] >= 0
            )
        )

        print(
            f"\nDirection accuracy: "
            f"{year_data['correct'].mean():.4f}"
        )
    print()
    print("================================")
    print("      SIGNAL STRENGTH ANALYSIS")
    print("================================")

    # Absolute prediction magnitude.
    results["abs_prediction"] = (
        results["prediction"].abs()
    )

    bins = [
        0.0000,
        0.0010,
        0.0025,
        0.0050,
        0.0100,
        float("inf")
    ]

    labels = [
        "0.00-0.10%",
        "0.10-0.25%",
        "0.25-0.50%",
        "0.50-1.00%",
        "1.00%+"
    ]

    results["signal_strength"] = pd.cut(
        results["abs_prediction"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    grouped = (
        results
        .groupby(
            "signal_strength",
            observed=True
        )
        .agg(
            predictions=(
                "actual_pct",
                "count"
            ),
            average_prediction=(
                "prediction_pct",
                "mean"
            ),
            average_actual=(
                "actual_pct",
                "mean"
            ),
            actual_std=(
                "actual_pct",
                "std"
            )
        )
    )

    print()
    print(
        grouped.to_string()
    )

    print()
    print("================================")
    print("       DIRECTIONAL SIGNAL")
    print("================================")

    # Does the sign of the prediction
    # correspond to the actual return?

    results["correct"] = (
        (
            results["prediction"] >= 0
        )
        ==
        (
            results["actual_pct"] >= 0
        )
    )

    print(
        f"\nOverall directional accuracy: "
        f"{results['correct'].mean():.4f}"
    )

    # Strongest 10% of predictions.
    threshold = results[
        "abs_prediction"
    ].quantile(0.90)

    strong = results[
        results["abs_prediction"] >= threshold
    ]

    print(
        f"\nTop 10% signal threshold: "
        f"{threshold * 100:.4f}%"
    )

    print(
        f"Strong-signal samples: "
        f"{len(strong)}"
    )

    print(
        f"Strong-signal direction accuracy: "
        f"{strong['correct'].mean():.4f}"
    )

    print(
        f"Strong-signal average actual return: "
        f"{strong['actual_pct'].mean():.4f}%"
    )


if __name__ == "__main__":
    main()