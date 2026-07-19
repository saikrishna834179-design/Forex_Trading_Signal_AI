import pandas as pd

# Load full technical signal data
technical_df = pd.read_csv("forex_signals_output.csv")

# Load ML predictions (only last test-period rows)
ml_df = pd.read_csv("forex_ml_predictions.csv")

# Merge ML prediction columns into technical dataset using DateTime
technical_df["DateTime"] = pd.to_datetime(technical_df["DateTime"])
ml_df["DateTime"] = pd.to_datetime(ml_df["DateTime"])

df = technical_df.merge(
    ml_df[["DateTime", "ML_Signal", "ML_Probability_Up"]],
    on="DateTime",
    how="left"
)

# For rows before ML test period, mark as unavailable
df["ML_Signal"] = df["ML_Signal"].fillna("NO_ML_SIGNAL")
df["ML_Probability_Up"] = df["ML_Probability_Up"].fillna(0.5)

# Final AI Signal
df["FinalAISignal"] = "HOLD"

# Strong agreement between Technical and ML
df.loc[
    (df["TechnicalSignal"].isin(["BUY", "STRONG_BUY"])) &
    (df["ML_Signal"] == "BUY"),
    "FinalAISignal"
] = "BUY"

df.loc[
    (df["TechnicalSignal"].isin(["SELL", "STRONG_SELL"])) &
    (df["ML_Signal"] == "SELL"),
    "FinalAISignal"
] = "SELL"

# Strong technical signal can still be used even if ML is unavailable
df.loc[
    (df["TechnicalSignal"] == "STRONG_BUY") &
    (df["ML_Signal"] == "NO_ML_SIGNAL"),
    "FinalAISignal"
] = "STRONG_BUY"

df.loc[
    (df["TechnicalSignal"] == "STRONG_SELL") &
    (df["ML_Signal"] == "NO_ML_SIGNAL"),
    "FinalAISignal"
] = "STRONG_SELL"

# Confidence score
df["SignalConfidence"] = 50

df.loc[df["TechnicalSignal"].isin(["STRONG_BUY", "STRONG_SELL"]), "SignalConfidence"] = 75

df.loc[
    (
        ((df["TechnicalSignal"].isin(["BUY", "STRONG_BUY"])) & (df["ML_Signal"] == "BUY"))
        |
        ((df["TechnicalSignal"].isin(["SELL", "STRONG_SELL"])) & (df["ML_Signal"] == "SELL"))
    ),
    "SignalConfidence"
] = 90

print(
    df[
        [
            "DateTime",
            "Close",
            "TechnicalSignal",
            "ML_Signal",
            "ML_Probability_Up",
            "FinalAISignal",
            "SignalConfidence"
        ]
    ].tail(25)
)

print("\nFinal AI Signal Counts:")
print(df["FinalAISignal"].value_counts())

df.to_csv("forex_final_ai_signals.csv", index=False)

print("\nDone! forex_final_ai_signals.csv created.")