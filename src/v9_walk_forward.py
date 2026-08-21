import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score


DATA_FILE = Path("data/market_features_v9.csv")


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
        column
        for column in data.columns
        if column not in [
            "Date",
            "target"
        ]
    ]

    print()
    print("================================")
    print("        V9 WALK-FORWARD")
    print("================================")

    results = []

    # Test each year independently.
    for test_year in range(2020, 2027):

        train = data[
            data["Date"].dt.year < test_year
        ]

        test = data[
            data["Date"].dt.year == test_year
        ]

        if len(train) == 0 or len(test) == 0:
            continue

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000
                )
            )
        ])

        print(
            f"\nTraining through "
            f"{test_year - 1}..."
        )

        model.fit(
            train[FEATURES],
            train["target"]
        )

        predictions = model.predict(
            test[FEATURES]
        )

        accuracy = accuracy_score(
            test["target"],
            predictions
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                test["target"],
                predictions
            )
        )

        results.append({
            "year": test_year,
            "accuracy": accuracy,
            "balanced_accuracy":
                balanced_accuracy,
            "samples": len(test)
        })

        print(
            f"Test year: {test_year}"
        )

        print(
            f"Samples: {len(test)}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Balanced accuracy: "
            f"{balanced_accuracy:.4f}"
        )

    results = pd.DataFrame(results)

    print()
    print("================================")
    print("            SUMMARY")
    print("================================")

    print()

    print(
        results.to_string(
            index=False
        )
    )

    print()

    print(
        f"Average accuracy: "
        f"{results['accuracy'].mean():.4f}"
    )

    print(
        f"Average balanced accuracy: "
        f"{results['balanced_accuracy'].mean():.4f}"
    )


if __name__ == "__main__":
    main()