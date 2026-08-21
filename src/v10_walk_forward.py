import pandas as pd

from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


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

    results = []

    print()
    print("================================")
    print("    V10 RETURN WALK-FORWARD")
    print("================================")

    for test_year in range(2020, 2027):

        train = data[
            data["Date"].dt.year < test_year
        ]

        test = data[
            data["Date"].dt.year == test_year
        ]

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

        print(
            f"\nTraining through "
            f"{test_year - 1}..."
        )

        model.fit(
            train[FEATURES],
            train["next_day_return"]
        )

        predictions = model.predict(
            test[FEATURES]
        )

        actual = test[
            "next_day_return"
        ]

        mae = mean_absolute_error(
            actual,
            predictions
        )

        rmse = (
            mean_squared_error(
                actual,
                predictions
            ) ** 0.5
        )

        if (
            actual.nunique() > 1
            and len(set(predictions)) > 1
        ):
            correlation = pearsonr(
                actual,
                predictions
            )[0]
        else:
            correlation = 0.0

        direction_accuracy = (
            (predictions >= 0)
            ==
            (actual >= 0)
        ).mean()

        results.append({
            "year": test_year,
            "MAE": mae,
            "RMSE": rmse,
            "correlation": correlation,
            "direction_accuracy":
                direction_accuracy,
            "avg_predicted_return":
                predictions.mean(),
            "avg_actual_return":
                actual.mean(),
            "samples": len(test)
        })

        print(
            f"Test year: {test_year}"
        )

        print(
            f"MAE: {mae:.6f}"
        )

        print(
            f"RMSE: {rmse:.6f}"
        )

        print(
            f"Correlation: {correlation:.4f}"
        )

        print(
            f"Direction accuracy: "
            f"{direction_accuracy:.4f}"
        )

        print(
            f"Avg predicted return: "
            f"{predictions.mean():.6f}"
        )

        print(
            f"Avg actual return: "
            f"{actual.mean():.6f}"
        )

    results = pd.DataFrame(results)

    print()
    print("================================")
    print("             SUMMARY")
    print("================================")

    print()

    print(
        results.to_string(
            index=False
        )
    )

    print()

    print(
        f"Average MAE: "
        f"{results['MAE'].mean():.6f}"
    )

    print(
        f"Average RMSE: "
        f"{results['RMSE'].mean():.6f}"
    )

    print(
        f"Average correlation: "
        f"{results['correlation'].mean():.4f}"
    )

    print(
        f"Average direction accuracy: "
        f"{results['direction_accuracy'].mean():.4f}"
    )


if __name__ == "__main__":
    main()