import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


DATA_FILE = Path("data/market_features_v1.csv")


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
        if c not in [
            "Date",
            "future_5d_return"
        ]
    ]

    yearly_results = []

    for test_year in range(2020, 2027):

        train = data[
            data["Date"].dt.year < test_year
        ]

        test = data[
            data["Date"].dt.year == test_year
        ].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        print()
        print(f"Training through {test_year - 1}...")
        print(f"Test year: {test_year}")
        print(f"Samples: {len(test)}")

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

        predictions = model.predict(
            test[FEATURES]
        )

        actual = test[
            "future_5d_return"
        ].values

        mae = mean_absolute_error(
            actual,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predictions
            )
        )

        correlation = pearsonr(
            actual,
            predictions
        )[0]

        direction_accuracy = np.mean(
            (predictions >= 0)
            ==
            (actual >= 0)
        )

        avg_prediction = np.mean(
            predictions
        )

        avg_actual = np.mean(
            actual
        )

        print(
            f"MAE: {mae:.6f}"
        )

        print(
            f"RMSE: {rmse:.6f}"
        )

        print(
            f"Correlation: "
            f"{correlation:.4f}"
        )

        print(
            f"Direction accuracy: "
            f"{direction_accuracy:.4f}"
        )

        print(
            f"Avg predicted 5-day return: "
            f"{avg_prediction:.4%}"
        )

        print(
            f"Avg actual 5-day return: "
            f"{avg_actual:.4%}"
        )

        yearly_results.append({
            "year": test_year,
            "MAE": mae,
            "RMSE": rmse,
            "correlation": correlation,
            "direction_accuracy":
                direction_accuracy,
            "avg_prediction":
                avg_prediction,
            "avg_actual":
                avg_actual,
            "samples": len(test)
        })

    results = pd.DataFrame(
        yearly_results
    )

    print()
    print("================================")
    print("             SUMMARY")
    print("================================")

    print(
        results.to_string(
            index=False,
            formatters={
                "MAE": "{:.6f}".format,
                "RMSE": "{:.6f}".format,
                "correlation":
                    "{:.4f}".format,
                "direction_accuracy":
                    "{:.4f}".format,
                "avg_prediction":
                    "{:.4%}".format,
                "avg_actual":
                    "{:.4%}".format
            }
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