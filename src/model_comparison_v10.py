import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    y_train = train["next_day_return"]

    X_validation = validation[FEATURES]
    y_validation = validation["next_day_return"]

    print()
    print("================================")
    print("      V10 RETURN MODEL")
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

    models = {

        "Ridge Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(alpha=10.0)
            )
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            loss="huber",
            random_state=42
        )
    }

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

        mae = mean_absolute_error(
            y_validation,
            predictions
        )

        rmse = mean_squared_error(
            y_validation,
            predictions
        ) ** 0.5

        correlation = pearsonr(
            y_validation,
            predictions
        )[0]

        actual_direction = (
            y_validation >= 0
        )

        predicted_direction = (
            predictions >= 0
        )

        direction_accuracy = (
            actual_direction ==
            predicted_direction
        ).mean()

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "correlation": correlation,
            "direction_accuracy":
                direction_accuracy
        })

        print(
            f"MAE: "
            f"{mae:.6f}"
        )

        print(
            f"RMSE: "
            f"{rmse:.6f}"
        )

        print(
            f"Correlation: "
            f"{correlation:.4f}"
        )

        print(
            f"Direction accuracy: "
            f"{direction_accuracy:.4f}"
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


if __name__ == "__main__":
    main()