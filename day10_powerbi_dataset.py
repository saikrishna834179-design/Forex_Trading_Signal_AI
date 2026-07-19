import pandas as pd

# Load final AI signals
signals = pd.read_csv("forex_final_ai_signals.csv")

# Load original backtest output
backtest = pd.read_csv("forex_backtest_output.csv")

# Load filtered backtest output
filtered = pd.read_csv("forex_filtered_backtest.csv")

# Convert DateTime
signals["DateTime"] = pd.to_datetime(signals["DateTime"])
backtest["DateTime"] = pd.to_datetime(backtest["DateTime"])
filtered["DateTime"] = pd.to_datetime(filtered["DateTime"])

# Keep useful original-strategy performance columns
backtest_cols = [
    "DateTime",
    "Position",
    "NetPnL",
    "CumulativePnL",
    "Drawdown"
]

backtest = backtest[backtest_cols]

# Keep useful filtered-strategy performance columns
filtered_cols = [
    "DateTime",
    "FilteredPosition",
    "FilteredNetPnL",
    "FilteredCumulativePnL",
    "FilteredDrawdown"
]

filtered = filtered[filtered_cols]

# Merge all project outputs into one dashboard dataset
powerbi_df = signals.merge(backtest, on="DateTime", how="left")

powerbi_df = powerbi_df.merge(
    filtered,
    on="DateTime",
    how="left"
)

# Create dashboard-friendly columns
powerbi_df["Date"] = powerbi_df["DateTime"].dt.date
powerbi_df["Hour"] = powerbi_df["DateTime"].dt.hour
powerbi_df["Month"] = powerbi_df["DateTime"].dt.month
powerbi_df["Year"] = powerbi_df["DateTime"].dt.year

# Risk category based on ATR
powerbi_df["RiskLevel"] = "Medium"
powerbi_df.loc[powerbi_df["ATR_14"] < powerbi_df["ATR_14"].median(), "RiskLevel"] = "Low"
powerbi_df.loc[
    powerbi_df["ATR_14"] > powerbi_df["ATR_14"].quantile(0.75),
    "RiskLevel"
] = "High"

# Fill missing values
powerbi_df["ML_Signal"] = powerbi_df["ML_Signal"].fillna("NO_ML_SIGNAL")
powerbi_df["FinalAISignal"] = powerbi_df["FinalAISignal"].fillna("HOLD")
powerbi_df["SignalConfidence"] = powerbi_df["SignalConfidence"].fillna(50)

# Save Power BI dataset
powerbi_df.to_csv("forex_powerbi_master.csv", index=False)

print("Power BI Master Dataset Created!")
print("Rows:", len(powerbi_df))
print("Columns:", len(powerbi_df.columns))

print("\nImportant Columns:")
print(powerbi_df.columns.tolist())

print("\nFirst 5 rows:")
print(powerbi_df.head())