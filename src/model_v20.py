import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score


DATA_FILE = Path("data/market_features_v19.csv")

TARGET = "future_5d_return"


def main():

    print("Loading V20 data...")

    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # CREATE DIRECTION TARGET
    # --------------------------------------------------

    data["direction"] = (
        data[TARGET] >= 0
    ).astype(int)

    FEATURES = [
        c for c in data.columns
        if c not in [
            "Date",
            "target",
            "future_1d_return",
            "future_3d_return",
            "future_5d_return",
            "future_10d_return",
            "direction",
        ]
    ]

    print()
    print(f"Rows: {len(data)}")
    print(f"Features: {len(FEATURES)}")

    print(
        f"Date range: "
        f"{data['Date'].min().date()} → "
        f"{data['Date'].max().date()}"
    )

    print()
    print("Direction distribution:")

    print(
        data["direction"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    yearly_results = []

    # --------------------------------------------------
    # WALK-FORWARD TEST
    # --------------------------------------------------

    for test_year in range(2020, 2027):

        train = data[
            data["Date"].dt.year < test_year
        ].copy()

        test = data[
            data["Date"].dt.year == test_year
        ].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        print()
        print("=" * 50)
        print(f"Training through {test_year - 1}")
        print(f"Test year: {test_year}")
        print(f"Samples: {len(test)}")
        print("=" * 50)

        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            loss="log_loss",
            random_state=42
        )

        model.fit(
            train[FEATURES],
            train["direction"]
        )

        predictions = model.predict(
            test[FEATURES]
        )

        probabilities = model.predict_proba(
            test[FEATURES]
        )[:, 1]

        actual = test["direction"].values

        accuracy = accuracy_score(
            actual,
            predictions
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                actual,
                predictions
            )
        )

        # Confidence = distance from 50%
        confidence = (
            np.abs(probabilities - 0.5)
            * 2
        )

        # High confidence = top 20%
        confidence_cutoff = np.quantile(
            confidence,
            0.80
        )

        strong = (
            confidence >= confidence_cutoff
        )

        if strong.sum() > 0:
            strong_accuracy = accuracy_score(
                actual[strong],
                predictions[strong]
            )
        else:
            strong_accuracy = 0.0

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Balanced accuracy: "
            f"{balanced_accuracy:.4f}"
        )

        print(
            f"Average UP probability: "
            f"{probabilities.mean():.4f}"
        )

        print(
            f"Top 20% confidence cutoff: "
            f"{confidence_cutoff:.4f}"
        )

        print(
            f"Strong-signal samples: "
            f"{strong.sum()}"
        )

        print(
            f"Strong-signal accuracy: "
            f"{strong_accuracy:.4f}"
        )

        yearly_results.append({
            "year": test_year,
            "accuracy": accuracy,
            "balanced_accuracy":
                balanced_accuracy,
            "strong_accuracy":
                strong_accuracy,
            "samples": len(test)
        })

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    results = pd.DataFrame(
        yearly_results
    )

    print()
    print("=" * 50)
    print("              V20 SUMMARY")
    print("=" * 50)

    print(
        results.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.4f}".format,
                "balanced_accuracy":
                    "{:.4f}".format,
                "strong_accuracy":
                    "{:.4f}".format,
            }
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

    print(
        f"Average strong-signal accuracy: "
        f"{results['strong_accuracy'].mean():.4f}"
    )


if __name__ == "__main__":
    main()