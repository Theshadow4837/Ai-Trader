import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


DATA_FILE = Path("data/SPY_features.csv")
MODEL_FILE = Path("models/v2_logistic_model.joblib")

FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "distance_sma10",
    "distance_sma50",
    "volatility_10d",
    "volatility_20d",
    "volume_change",
    "volume_ratio",
    "daily_range",
]

TARGET = "target"


def train_model():

    # Load data
    data = pd.read_csv(DATA_FILE)

    # Make sure the data is chronological
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # 80/20 chronological split
    split_index = int(len(data) * 0.8)

    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    print(f"\nTraining period:")
    print(f"{train['Date'].iloc[0].date()} → {train['Date'].iloc[-1].date()}")

    print(f"\nTesting period:")
    print(f"{test['Date'].iloc[0].date()} → {test['Date'].iloc[-1].date()}")

    # Model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
       ("classifier", LogisticRegression(
    max_iter=1000
))
    ])

    # Train
    print("\nTraining model...")

    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    # Save model
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved to {MODEL_FILE}")


if __name__ == "__main__":
    train_model()