import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("forex_price_data.csv")

# Convert DateTime and sort
df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values(["PairName", "DateTime"]).reset_index(drop=True)

# -----------------------------
# SMA indicators
# -----------------------------
df["SMA_10"] = df["Close"].rolling(10).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()

df["Trend"] = "HOLD"
df.loc[df["SMA_10"] > df["SMA_50"], "Trend"] = "BUY"
df.loc[df["SMA_10"] < df["SMA_50"], "Trend"] = "SELL"

# -----------------------------
# RSI 14
# -----------------------------
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI_14"] = 100 - (100 / (1 + rs))

df["RSI_Signal"] = "NEUTRAL"
df.loc[df["RSI_14"] < 30, "RSI_Signal"] = "BUY"
df.loc[df["RSI_14"] > 70, "RSI_Signal"] = "SELL"

# -----------------------------
# MACD
# -----------------------------
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["MACD_Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal_Line"]

df["MACD_Signal"] = "NEUTRAL"
df.loc[df["MACD"] > df["MACD_Signal_Line"], "MACD_Signal"] = "BUY"
df.loc[df["MACD"] < df["MACD_Signal_Line"], "MACD_Signal"] = "SELL"

# -----------------------------
# Bollinger Bands
# -----------------------------
df["BB_Middle"] = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()

df["BB_Upper"] = df["BB_Middle"] + (2 * bb_std)
df["BB_Lower"] = df["BB_Middle"] - (2 * bb_std)

df["BB_Signal"] = "NEUTRAL"
df.loc[df["Close"] < df["BB_Lower"], "BB_Signal"] = "BUY"
df.loc[df["Close"] > df["BB_Upper"], "BB_Signal"] = "SELL"

# -----------------------------
# ATR 14
# -----------------------------
high_low = df["High"] - df["Low"]
high_close = (df["High"] - df["Close"].shift()).abs()
low_close = (df["Low"] - df["Close"].shift()).abs()

true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["ATR_14"] = true_range.rolling(14).mean()

# -----------------------------
# Composite Technical Signal
# -----------------------------
signal_map = {
    "BUY": 1,
    "SELL": -1,
    "HOLD": 0,
    "NEUTRAL": 0
}

df["Trend_Score"] = df["Trend"].map(signal_map)
df["RSI_Score"] = df["RSI_Signal"].map(signal_map)
df["MACD_Score"] = df["MACD_Signal"].map(signal_map)
df["BB_Score"] = df["BB_Signal"].map(signal_map)

df["TechnicalScore"] = (
    df["Trend_Score"]
    + df["RSI_Score"]
    + df["MACD_Score"]
    + df["BB_Score"]
)

df["TechnicalSignal"] = "HOLD"

df.loc[df["TechnicalScore"] >= 2, "TechnicalSignal"] = "STRONG_BUY"
df.loc[df["TechnicalScore"] == 1, "TechnicalSignal"] = "BUY"
df.loc[df["TechnicalScore"] == -1, "TechnicalSignal"] = "SELL"
df.loc[df["TechnicalScore"] <= -2, "TechnicalSignal"] = "STRONG_SELL"

# -----------------------------
# Show latest rows
# -----------------------------
print(
    df[
        [
            "DateTime",
            "Close",
            "Trend",
            "RSI_Signal",
            "MACD_Signal",
            "BB_Signal",
            "TechnicalScore",
            "TechnicalSignal",
            "ATR_14"
        ]
    ].tail(25)
)

print("\nTechnical Signal Counts:")
print(df["TechnicalSignal"].value_counts())

df.to_csv("forex_signals_output.csv", index=False)

print("\nDone! forex_signals_output.csv created.")