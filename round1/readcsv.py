import pandas as pd
import matplotlib.pyplot as plt
import jsonpickle

df_round1_day_neg2 = pd.read_csv("./IMC testing/round1/prices_round_1_day_-2.csv", delimiter=';')
df_round1_day_neg1 = pd.read_csv("./IMC testing/round1/prices_round_1_day_-1.csv", delimiter=';')
df_round1_day_zero = pd.read_csv("./IMC testing/round1/prices_round_1_day_0.csv", delimiter=';')

def split_df(df):
    dfA = df[df["product"] == "AMETHYSTS"]
    dfS = df[df["product"] == "STARFRUIT"]
    return dfA, dfS

dfAneg2, dfSneg2 = split_df(df_round1_day_neg2)
dfAneg1, dfSneg1 = split_df(df_round1_day_neg1)
dfAzero, dfSzero = split_df(df_round1_day_zero)

dfAneg2 = dfAneg2[dfAneg2["timestamp"] < 1000] # limiting to 50 iterations for graphing
dfSneg2 = dfSneg2[dfSneg2["timestamp"] < 100000] # limiting to 500 iterations for graphing

mid_price_rolling_avg = dfSneg2["mid_price"].rolling(window = 20).mean()
mid_price_rolling_avg_long = dfSneg2["mid_price"].rolling(window = 100).mean()
dfSneg2["rollingmid"] = mid_price_rolling_avg

rolling_std = dfSneg2["mid_price"].rolling(window = 20).std()
long_rolling_std = dfSneg2["mid_price"].rolling(window = 100).std()
dfSneg2["rollingstd"] = rolling_std

dfSneg2["upperband"] = dfSneg2["rollingmid"] + 2 * dfSneg2["rollingstd"]
dfSneg2["lowerband"] = dfSneg2["rollingmid"] - 2 * dfSneg2["rollingstd"]

dfSneg2[["bid_price_1", "bid_price_2", "bid_price_3"]].fillna(0, inplace=True)
dfSneg2[["ask_price_1", "ask_price_2", "ask_price_3"]].fillna(100000, inplace = True)

dfBelowLower = dfSneg2[dfSneg2["ask_price_1"] < dfSneg2["lowerband"]]
dfAboveUpper = dfSneg2[dfSneg2["bid_price_1"] > dfSneg2["upperband"]]

dfBelowLower2 = dfSneg2[dfSneg2["ask_price_2"] < dfSneg2["lowerband"]]
dfAboveUpper2 = dfSneg2[dfSneg2["bid_price_2"] > dfSneg2["upperband"]]

dfBelowLower3 = dfSneg2[dfSneg2["ask_price_3"] < dfSneg2["lowerband"]]
dfAboveUpper3 = dfSneg2[dfSneg2["bid_price_3"] > dfSneg2["upperband"]]

rowsLower, colsLower = dfBelowLower.shape
rowsHigher, colsHigher = dfAboveUpper.shape

rowsLower2, colsLower2 = dfBelowLower2.shape
rowsHigher2, colsHigher2 = dfAboveUpper2.shape

rowsLower3, colsLower3 = dfBelowLower3.shape
rowsHigher3, colsHigher3 = dfAboveUpper3.shape

print(f"Number of order flags {rowsLower + rowsHigher}")
print(f"Number of order flags {rowsLower2 + rowsHigher2}")
print(f"Number of order flags {rowsLower3 + rowsHigher3}")

plt.scatter(dfAboveUpper["timestamp"], dfAboveUpper["mid_price"], color = "green", s = 100)
plt.scatter(dfBelowLower["timestamp"], dfBelowLower["mid_price"], color = "green", s = 100)
plt.scatter(dfSneg2["timestamp"], dfSneg2["mid_price"] , alpha = 0.2, color = "red")

plt.plot(dfSneg2["timestamp"], mid_price_rolling_avg, label = 'Rolling Average')
plt.plot(dfSneg2["timestamp"], mid_price_rolling_avg_long, label = 'Long Rolling Average', color = "black")

# plt.plot(dfSneg2["timestamp"], dfSneg2["lowerband"], label = "Lower band")
# plt.plot(dfSneg2["timestamp"], dfSneg2["upperband"], label = "Upper band")

plt.xlabel("timestamp")
plt.ylabel("mid_price")
plt.title("day -2 mid price over 50 iterations")

plt.show()

print(dfSneg2.iloc[1])