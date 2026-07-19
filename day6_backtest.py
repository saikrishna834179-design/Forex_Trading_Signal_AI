import pandas as pd
import numpy as np

# Load signal output
df = pd.read_csv("forex_signals_output.csv")

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Next candle return
df["Next_Close"] = df["Close"].shift(-1)
df["Price_Change"] = df["Next_Close"] - df["Close"]

# Convert signal into trade direction
df["Position"] = 0

df.loc[df["TechnicalSignal"].isin(["BUY", "STRONG_BUY"]), "Position"] = 1
df.loc[df["TechnicalSignal"].isin(["SELL", "STRONG_SELL"]), "Position"] = -1

# Profit/Loss for one standard demo position
lot_size = 100000

df["TradePnL"] = df["Position"] * df["Price_Change"] * lot_size

# Transaction cost using spread
df["TransactionCost"] = np.where(
    df["Position"] != 0,
    df["Spread"] * 0.0001 * lot_size,
    0
)

df["NetPnL"] = df["TradePnL"] - df["TransactionCost"]

# Cumulative P&L
df["CumulativePnL"] = df["NetPnL"].cumsum()

# Running peak and drawdown
df["RunningPeak"] = df["CumulativePnL"].cummax()
df["Drawdown"] = df["CumulativePnL"] - df["RunningPeak"]

# Remove last row because it has no next price
df = df.dropna(subset=["Next_Close"])

# Performance metrics
total_trades = (df["Position"] != 0).sum()
winning_trades = (df["NetPnL"] > 0).sum()
losing_trades = (df["NetPnL"] < 0).sum()

win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
total_pnl = df["NetPnL"].sum()
max_drawdown = df["Drawdown"].min()

print("BACKTEST RESULTS")
print("-" * 30)
print("Total Trades:", total_trades)
print("Winning Trades:", winning_trades)
print("Losing Trades:", losing_trades)
print("Win Rate:", round(win_rate, 2), "%")
print("Total Net PnL:", round(total_pnl, 2))
print("Maximum Drawdown:", round(max_drawdown, 2))

# Save backtest output
df.to_csv("forex_backtest_output.csv", index=False)

print("\nDone! forex_backtest_output.csv created.")