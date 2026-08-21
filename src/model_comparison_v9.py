import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score


DATA_FILE = Path("data/market_features_v9.csv")


def make_models():

    return {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            random_state=42
        )
    }


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
            "target"
        ]
    ]

    train = data[
        data["Date"].dt.year <= 2021
    ]

    validation = data[
        data["Date"].dt.year.between(
            2022,
            2024
        )
    ]

    X_train = train[FEATURES]
    y_train = train["target"]

    X_validation = validation[FEATURES]
    y_validation = validation["target"]

    print()
    print("================================")
    print("       V9 MODEL COMPARISON")
    print("================================")

    print(
        f"\nFeatures used: "
        f"{len(FEATURES)}"
    )

    print(
        f"Training rows: "
        f"{len(train)}"
    )

    print(
        f"Validation rows: "
        f"{len(validation)}"
    )

    print(
        f"\nTraining period: "
        f"{train['Date'].min().date()} → "
        f"{train['Date'].max().date()}"
    )

    print(
        f"Validation period: "
        f"{validation['Date'].min().date()} → "
        f"{validation['Date'].max().date()}"
    )

    models = make_models()

    results = []

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_validation
        )

        accuracy = accuracy_score(
            y_validation,
            predictions
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_validation,
                predictions
            )
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "balanced_accuracy":
                balanced_accuracy
        })

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
    print("            RESULTS")
    print("================================")

    print(
        results.to_string(
            index=False
        )
    )

    best = results.sort_values(
        "balanced_accuracy",
        ascending=False
    ).iloc[0]

    print()

    print(
        f"Best model: "
        f"{best['model']}"
    )

    print(
        f"Best balanced accuracy: "
        f"{best['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()