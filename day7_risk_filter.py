import pandas as pd
import numpy as np

df = pd.read_csv("forex_backtest_output.csv")

# Trade only when there is strong agreement
df["FilteredPosition"] = 0

df.loc[df["TechnicalSignal"] == "STRONG_BUY", "FilteredPosition"] = 1
df.loc[df["TechnicalSignal"] == "STRONG_SELL", "FilteredPosition"] = -1

# Recalculate P&L with only strong signals
lot_size = 100000

df["FilteredTradePnL"] = (
    df["FilteredPosition"] * df["Price_Change"] * lot_size
)

df["FilteredTransactionCost"] = np.where(
    df["FilteredPosition"] != 0,
    df["Spread"] * 0.0001 * lot_size,
    0
)

df["FilteredNetPnL"] = (
    df["FilteredTradePnL"] - df["FilteredTransactionCost"]
)

df["FilteredCumulativePnL"] = df["FilteredNetPnL"].cumsum()

df["FilteredRunningPeak"] = df["FilteredCumulativePnL"].cummax()

df["FilteredDrawdown"] = (
    df["FilteredCumulativePnL"] - df["FilteredRunningPeak"]
)

# Metrics
total_trades = (df["FilteredPosition"] != 0).sum()
winning_trades = (
    (df["FilteredPosition"] != 0) &
    (df["FilteredNetPnL"] > 0)
).sum()

losing_trades = (
    (df["FilteredPosition"] != 0) &
    (df["FilteredNetPnL"] < 0)
).sum()

win_rate = (winning_trades / total_trades * 100) if total_trades else 0

print("FILTERED BACKTEST RESULTS")
print("-" * 35)
print("Total Trades:", total_trades)
print("Winning Trades:", winning_trades)
print("Losing Trades:", losing_trades)
print("Win Rate:", round(win_rate, 2), "%")
print("Total Net PnL:", round(df["FilteredNetPnL"].sum(), 2))
print("Maximum Drawdown:", round(df["FilteredDrawdown"].min(), 2))

df.to_csv("forex_filtered_backtest.csv", index=False)

print("\nDone! forex_filtered_backtest.csv created.")
