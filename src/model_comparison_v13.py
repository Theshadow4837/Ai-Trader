import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


DATA_FILE = Path("data/market_features_v13.csv")


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

    train = data[data["Date"].dt.year <= 2021]
    validation = data[
        data["Date"].dt.year.between(2022, 2024)
    ]

    X_train = train[FEATURES]
    y_train = train["future_5d_return"]

    X_val = validation[FEATURES]
    y_val = validation["future_5d_return"]

    print()
    print("================================")
    print("      V13 5-DAY RETURN MODEL")
    print("================================")

    print(f"\nFeatures used: {len(FEATURES)}")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")

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
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
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

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        mae = mean_absolute_error(
            y_val,
            predictions
        )

        rmse = mean_squared_error(
            y_val,
            predictions
        ) ** 0.5

        correlation = pearsonr(
            y_val,
            predictions
        )[0]

        direction_accuracy = (
            (predictions >= 0)
            ==
            (y_val >= 0)
        ).mean()

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "correlation": correlation,
            "direction_accuracy":
                direction_accuracy
        })

        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"Correlation: {correlation:.4f}")
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
        results.to_string(index=False)
    )


if __name__ == "__main__":
    main()