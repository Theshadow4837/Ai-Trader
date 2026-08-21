import pandas as pd
import joblib

from sklearn.metrics import accuracy_score


DATA_FILE = "data/SPY_features.csv"
MODEL_FILE = "models/logistic_model.joblib"

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

data = pd.read_csv(DATA_FILE)

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

split_index = int(len(data) * 0.8)

test = data.iloc[split_index:].copy()

X_test = test[FEATURES]
y_test = test["target"]

model = joblib.load(MODEL_FILE)

predictions = model.predict(X_test)

# AI
ai_accuracy = accuracy_score(y_test, predictions)

# Always predict UP
baseline_predictions = [1] * len(y_test)
baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("===== MODEL COMPARISON =====")

print(f"AI accuracy:       {ai_accuracy:.4f}")
print(f"Always-UP accuracy: {baseline_accuracy:.4f}")

print()

if ai_accuracy > baseline_accuracy:
    print("AI BEATS THE BASELINE")
else:
    print("AI DOES NOT BEAT THE BASELINE")