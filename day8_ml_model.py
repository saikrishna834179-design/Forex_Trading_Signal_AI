import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load signal dataset
df = pd.read_csv("forex_signals_output.csv")

# Create target:
# 1 = next candle close is higher
# 0 = next candle close is lower or equal
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Features used by the ML model
features = [
    "SMA_10",
    "SMA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal_Line",
    "MACD_Histogram",
    "BB_Upper",
    "BB_Lower",
    "ATR_14",
    "Volume",
    "Spread"
]

# Remove rows with missing indicator values
ml_df = df.dropna(subset=features).copy()

# Keep time order: first 80% training, last 20% testing
split_index = int(len(ml_df) * 0.8)

train_df = ml_df.iloc[:split_index]
test_df = ml_df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df["Target"]

X_test = test_df[features]
y_test = test_df["Target"]

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
test_df["ML_Prediction"] = model.predict(X_test)
test_df["ML_Probability_Up"] = model.predict_proba(X_test)[:, 1]

# Convert ML prediction to signal
test_df["ML_Signal"] = "SELL"
test_df.loc[test_df["ML_Prediction"] == 1, "ML_Signal"] = "BUY"

# Accuracy
accuracy = accuracy_score(y_test, test_df["ML_Prediction"])

print("RANDOM FOREST MODEL RESULTS")
print("-" * 35)
print("Training Rows:", len(train_df))
print("Testing Rows:", len(test_df))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, test_df["ML_Prediction"]))

# Feature importance
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Save ML predictions
test_df.to_csv("forex_ml_predictions.csv", index=False)

print("\nDone! forex_ml_predictions.csv created.")