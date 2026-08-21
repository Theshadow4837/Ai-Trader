import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)


# ============================================================
# V21.1
# Reward-weighted 3-class market classifier
# ============================================================

DATA_FILE = Path("data/market_features_v14.csv")

HOLD_THRESHOLD = 0.0020
# +/- 0.20%
#
# Below -0.20% = DOWN
# Between -0.20% and +0.20% = HOLD
# Above +0.20% = UP


def create_target(df):

    future = df["future_5d_return"]

    target = np.select(
        [
            future < -HOLD_THRESHOLD,
            future > HOLD_THRESHOLD
        ],
        [
            0,  # DOWN
            2   # UP
        ],
        default=1   # HOLD
    )

    return target.astype(int)


def create_sample_weights(df):

    """
    Give more importance to meaningful market moves.

    Tiny movements are noisy.

    Large correct/incorrect movements matter more.
    """

    returns = df["future_5d_return"].abs()

    weights = 1.0 + (
        returns / returns.median()
    )

    # Prevent extreme outliers from dominating training.
    weights = np.clip(
        weights,
        1.0,
        5.0
    )

    return weights.values


def strong_signal_probability(probabilities):

    """
    Confidence = probability of the predicted class.
    """

    return probabilities.max(axis=1)


def main():

    print("=" * 60)
    print("                 V21.1")
    print("        REWARD-WEIGHTED CLASSIFIER")
    print("=" * 60)

    data = pd.read_csv(DATA_FILE)

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------
    # --------------------------------------------------------

    FEATURES = [
    c for c in data.columns
    if c not in [
        "Date",
        "target",
        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return"
    ]
]
    print()
    print("FEATURE CHECK")
    print(f"Number of features: {len(FEATURES)}")

    for feature in FEATURES:
        print(f" - {feature}")

    print()
    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    data["target"] = create_target(data)

    print()
    print(f"Rows: {len(data)}")
    print(f"Features: {len(FEATURES)}")

    print()
    print("Target distribution:")

    counts = data["target"].value_counts(
        normalize=True
    ).sort_index()

    print(
        f"DOWN : {counts.get(0, 0):.2%}"
    )

    print(
        f"HOLD : {counts.get(1, 0):.2%}"
    )

    print(
        f"UP   : {counts.get(2, 0):.2%}"
    )

    yearly_results = []

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

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
        print("-" * 60)
        print(
            f"Training through {test_year - 1}"
        )
        print(
            f"Testing: {test_year}"
        )
        print(
            f"Samples: {len(test)}"
        )

        # ----------------------------------------------------
        # Reward weights
        # ----------------------------------------------------

        sample_weights = create_sample_weights(
            train
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            subsample=0.85,
            random_state=42
        )

        model.fit(
            train[FEATURES],
            train["target"],
            sample_weight=sample_weights
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(
            test[FEATURES]
        )

        probabilities = model.predict_proba(
            test[FEATURES]
        )

        confidence = strong_signal_probability(
            probabilities
        )

        actual = test["target"].values

        # ----------------------------------------------------
        # Normal accuracy
        # ----------------------------------------------------

        accuracy = accuracy_score(
            actual,
            predictions
        )

        balanced = balanced_accuracy_score(
            actual,
            predictions
        )

        # ----------------------------------------------------
        # Strong signal
        #
        # Only evaluate predictions where model confidence
        # is >= 60%.
        # ----------------------------------------------------

        strong_mask = confidence >= 0.60

        if strong_mask.sum() > 0:

            strong_accuracy = accuracy_score(
                actual[strong_mask],
                predictions[strong_mask]
            )

            strong_samples = int(
                strong_mask.sum()
            )

        else:

            strong_accuracy = np.nan
            strong_samples = 0

        # ----------------------------------------------------
        # Reward
        #
        # Correct UP/DOWN predictions receive the actual
        # magnitude of the move.
        #
        # Wrong directional predictions receive a penalty.
        #
        # HOLD gets zero directional reward.
        # ----------------------------------------------------

        actual_return = (
            test["future_5d_return"]
            .values
        )

        reward = np.zeros(
            len(test)
        )

        # Correct UP
        mask = (
            (predictions == 2)
            &
            (actual_return > 0)
        )

        reward[mask] = (
            actual_return[mask]
        )

        # Wrong UP
        mask = (
            (predictions == 2)
            &
            (actual_return < 0)
        )

        reward[mask] = (
            actual_return[mask]
        )

        # Correct DOWN
        mask = (
            (predictions == 0)
            &
            (actual_return < 0)
        )

        reward[mask] = (
            -actual_return[mask]
        )

        # Wrong DOWN
        mask = (
            (predictions == 0)
            &
            (actual_return > 0)
        )

        reward[mask] = (
            -actual_return[mask]
        )

        average_reward = reward.mean()

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Balanced accuracy: "
            f"{balanced:.4f}"
        )

        print(
            f"Strong accuracy: "
            f"{strong_accuracy:.4f}"
        )

        print(
            f"Strong samples: "
            f"{strong_samples}"
        )

        print(
            f"Average reward: "
            f"{average_reward:.6f}"
        )

        yearly_results.append({
            "year": test_year,
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "average_reward": average_reward,
            "strong_accuracy": strong_accuracy,
            "strong_samples": strong_samples,
            "samples": len(test)
        })

    # ========================================================
    # SUMMARY
    # ========================================================

    results = pd.DataFrame(
        yearly_results
    )

    print()
    print("=" * 60)
    print("                 V21.1 SUMMARY")
    print("=" * 60)

    print()

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
                    "{:.4f}".format
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

    print()
    print("=" * 60)
    print("V21.1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()