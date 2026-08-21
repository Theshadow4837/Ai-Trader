import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score


DATA_FILE = Path("data/market_features_v19.csv")

HORIZON = 5

# Minimum return considered a meaningful move.
REWARD_THRESHOLD = 0.002


def main():

    print("================================")
    print("        V21 REWARD MODEL")
    print("================================")

    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"])

    data = (
        data
        .sort_values("Date")
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Create the target
    #
    # +1 = meaningful positive return
    #  0 = roughly flat
    # -1 = meaningful negative return
    # --------------------------------------------------

    if "future_5d_return" not in data.columns:
        raise ValueError(
            "future_5d_return is missing from the dataset."
        )

    data["reward_target"] = np.select(
        [
            data["future_5d_return"] >= REWARD_THRESHOLD,
            data["future_5d_return"] <= -REWARD_THRESHOLD
        ],
        [
            1,
            -1
        ],
        default=0
    )

    data = data.dropna(
        subset=["future_5d_return"]
    ).reset_index(drop=True)

    FEATURES = [
        c for c in data.columns
        if c not in [
            "Date",
            "future_1d_return",
            "future_3d_return",
            "future_5d_return",
            "future_10d_return",
            "reward_target"
        ]
    ]

    print()
    print(f"Rows: {len(data)}")
    print(f"Features: {len(FEATURES)}")

    print()
    print("Reward distribution:")

    counts = data["reward_target"].value_counts(
        normalize=True
    ).sort_index()

    for label, percentage in counts.items():

        if label == -1:
            name = "NEGATIVE"
        elif label == 0:
            name = "FLAT"
        else:
            name = "POSITIVE"

        print(
            f"{name:>10}: "
            f"{percentage:.2%}"
        )

    # --------------------------------------------------
    # Walk-forward evaluation
    # --------------------------------------------------

    yearly_results = []

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
        print(
            f"Training through {test_year - 1}..."
        )
        print(
            f"Testing: {test_year}"
        )

        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=15,
            subsample=0.8,
            random_state=42
        )

        model.fit(
            train[FEATURES],
            train["reward_target"]
        )

        predictions = model.predict(
            test[FEATURES]
        )

        probabilities = model.predict_proba(
            test[FEATURES]
        )

        classes = model.classes_

        # Probability of positive / negative outcome.
        positive_probability = np.zeros(
            len(test)
        )

        negative_probability = np.zeros(
            len(test)
        )

        if 1 in classes:
            positive_probability = probabilities[
                :,
                list(classes).index(1)
            ]

        if -1 in classes:
            negative_probability = probabilities[
                :,
                list(classes).index(-1)
            ]

        actual = test[
            "reward_target"
        ].values

        accuracy = accuracy_score(
            actual,
            predictions
        )

        balanced_accuracy = balanced_accuracy_score(
            actual,
            predictions
        )

        # --------------------------------------------------
        # Reward calculation
        #
        # Correct prediction = +1
        # Wrong prediction   = -1
        # Flat prediction is rewarded when the actual
        # outcome is also flat.
        # --------------------------------------------------

        rewards = np.where(
            predictions == actual,
            1.0,
            -1.0
        )

        average_reward = rewards.mean()

        # Strong-confidence predictions.
        confidence = np.maximum(
            positive_probability,
            negative_probability
        )

        strong = confidence >= 0.60

        if strong.sum() > 0:

            strong_accuracy = np.mean(
                predictions[strong]
                ==
                actual[strong]
            )

        else:
            strong_accuracy = np.nan

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Balanced accuracy: "
            f"{balanced_accuracy:.4f}"
        )

        print(
            f"Average reward: "
            f"{average_reward:.4f}"
        )

        print(
            f"Strong-signal accuracy: "
            f"{strong_accuracy:.4f}"
        )

        print(
            f"Strong signals: "
            f"{strong.sum()}"
        )

        yearly_results.append({
            "year": test_year,
            "accuracy": accuracy,
            "balanced_accuracy":
                balanced_accuracy,
            "average_reward":
                average_reward,
            "strong_accuracy":
                strong_accuracy,
            "samples": len(test)
        })

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    results = pd.DataFrame(
        yearly_results
    )

    print()
    print("================================")
    print("          V21 SUMMARY")
    print("================================")

    print(
        results.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.4f}".format,
                "balanced_accuracy":
                    "{:.4f}".format,
                "average_reward":
                    "{:.4f}".format,
                "strong_accuracy":
                    lambda x:
                    "N/A"
                    if pd.isna(x)
                    else f"{x:.4f}"
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
        f"Average reward: "
        f"{results['average_reward'].mean():.4f}"
    )

    print(
        f"Average strong-signal accuracy: "
        f"{results['strong_accuracy'].mean():.4f}"
    )


if __name__ == "__main__":
    main()