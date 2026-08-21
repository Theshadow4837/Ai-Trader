import pandas as pd
import joblib


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

model = joblib.load(MODEL_FILE)

X_test = test[FEATURES]

# Probability of the next day being UP
probabilities = model.predict_proba(X_test)[:, 1]

test["prob_up"] = probabilities
test["prediction"] = (probabilities >= 0.5).astype(int)

print("===== PREDICTION ANALYSIS =====")

print(f"Total predictions: {len(test)}")

print(
    f"Predicted UP:   "
    f"{(test['prediction'] == 1).sum()}"
)

print(
    f"Predicted DOWN: "
    f"{(test['prediction'] == 0).sum()}"
)

print("\nPrediction probability statistics:")

print(
    test["prob_up"].describe()
)

print("\nMost confident predictions:")

print(
    test[
        ["Date", "Close", "prob_up", "target"]
    ]
    .sort_values("prob_up", ascending=False)
    .head(10)
    .to_string(index=False)
)