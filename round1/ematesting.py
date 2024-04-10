import pandas as pd
import matplotlib.pyplot as plt


def get_ema(list_of_mid_prices: list, period : int) -> float:
        list_to_series = pd.Series(list_of_mid_prices)
        return list_to_series.ewm(span=period, adjust=False).mean()

adjustment = 0.1

df_round1_day_neg2 = pd.read_csv("./IMC testing/round1/prices_round_1_day_-2.csv", delimiter=';')
df_round1_day_neg1 = pd.read_csv("./IMC testing/round1/prices_round_1_day_-1.csv", delimiter=';')
df_round1_day_zero = pd.read_csv("./IMC testing/round1/prices_round_1_day_0.csv", delimiter=';')

df_round1_day_neg2_trades = pd.read_csv("./IMC testing/round1/trades_round_1_day_-2_nn.csv", delimiter=';')

def split_df_trades(df):
    dfA = df[df["symbol"] == "AMETHYSTS"]
    dfS = df[df["symbol"] == "STARFRUIT"]
    return dfA, dfS

def split_df(df):
    dfA = df[df["product"] == "AMETHYSTS"]
    dfS = df[df["product"] == "STARFRUIT"]
    return dfA, dfS

dfAneg2, dfSneg2 = split_df(df_round1_day_neg2)
dfAneg2_trades, dfSneg2_trades = split_df_trades(df_round1_day_neg2_trades)

dfSneg2 = dfSneg2[dfSneg2["timestamp"] <= 50000] # limiting to 500 iterations for graphing

# Calculate exponential moving average (EMA)
ema_long = dfSneg2["mid_price"].ewm(adjust=False, span = 100).mean()
dfSneg2["ema_long"] = ema_long

# calc sma
sma_short = dfSneg2["mid_price"].rolling(window = 36).mean()
dfSneg2["sma_short"] = sma_short

dfSneg2Cross = dfSneg2[abs(dfSneg2["sma_short"] - dfSneg2["ema_long"]) < adjustment]
cross_price_trades = dfSneg2_trades[dfSneg2_trades["timestamp"].isin(dfSneg2Cross["timestamp"])]

# bands
rolling_std = dfSneg2["mid_price"].rolling(window = 10).std()
long_rolling_std = dfSneg2["mid_price"].rolling(window = 50).std()
dfSneg2["rollingstd"] = rolling_std

scale = 1.75
dfSneg2["upperband"] = dfSneg2["sma_short"] + scale * dfSneg2["rollingstd"]
dfSneg2["lowerband"] = dfSneg2["sma_short"] - scale * dfSneg2["rollingstd"]

dfBelowLower = dfSneg2[dfSneg2["ask_price_1"] < dfSneg2["lowerband"]]
dfAboveUpper = dfSneg2[dfSneg2["bid_price_1"] > dfSneg2["upperband"]]

# cmaps
cmap_above = plt.cm.Reds
cmap_below = plt.cm.Blues
cmap_cross = plt.cm.Greens

plt.scatter(dfAboveUpper["timestamp"], dfAboveUpper["bid_price_1"], c = dfAboveUpper["bid_volume_1"], s = 50, cmap=cmap_above)
plt.scatter(dfBelowLower["timestamp"], dfBelowLower["ask_price_1"], c = dfBelowLower["ask_volume_1"], s = 50, cmap=cmap_below)

# Plot the data and EMA curves
# plt.scatter(dfSneg2["timestamp"], dfSneg2["mid_price"], color="red", alpha=0.5, label="Mid Price")
plt.scatter(cross_price_trades["timestamp"], cross_price_trades["price"], c = cross_price_trades["quantity"], cmap=cmap_cross)
plt.plot(dfSneg2["timestamp"], ema_long, color="green", label="Long EMA (80)")
plt.plot(dfSneg2["timestamp"], sma_short, label = 'SMA')

# Add labels and title
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")
plt.title("Exponential Moving Average")
plt.colorbar(label='Quantity Traded')

# Add legend
plt.legend()

# Show plot
plt.show()