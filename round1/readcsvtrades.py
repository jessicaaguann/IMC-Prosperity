import pandas as pd
import matplotlib.pyplot as plt

df_round1_day_neg2 = pd.read_csv("./IMC testing/round1/trades_round_1_day_-2_nn.csv", delimiter=';')
df_round1_day_neg2_prices = pd.read_csv("./IMC testing/round1/prices_round_1_day_-2.csv", delimiter=';')

df_round1_day_neg1 = pd.read_csv("./IMC testing/round1/trades_round_1_day_-1_nn.csv", delimiter=';')
df_round1_day_zero = pd.read_csv("./IMC testing/round1/trades_round_1_day_0_nn.csv", delimiter=';')

def split_df(df):
    dfA = df[df["symbol"] == "AMETHYSTS"]
    dfS = df[df["symbol"] == "STARFRUIT"]
    return dfA, dfS

def split_df_prices(df):
    dfA = df[df["product"] == "AMETHYSTS"]
    dfS = df[df["product"] == "STARFRUIT"]
    return dfA, dfS

dfAneg2, dfSneg2 = split_df(df_round1_day_neg2)
print(dfSneg2.head(10))
dfAneg2_p, dfSneg2_p = split_df_prices(df_round1_day_neg2_prices)

count = dfAneg2["price"].value_counts()
print(count)

dfAneg2 = dfAneg2[dfAneg2["timestamp"] < 15000]
dfAneg2_p = dfAneg2_p[dfAneg2_p["timestamp"] < 15000]
dfAneg2_p = dfAneg2_p[dfAneg2_p["timestamp"].isin(dfAneg2["timestamp"])]

dfAneg2_p["mid_price"] = dfAneg2_p["mid_price"].round(0)


dfSneg2 = dfSneg2[dfSneg2["timestamp"] < 60000]
dfSneg2_p = dfSneg2_p[dfSneg2_p["timestamp"] < 60000]
dfSneg2_p = dfSneg2_p[dfSneg2_p["timestamp"].isin(dfSneg2["timestamp"])]

dfSneg2_p["mid_price"] = dfSneg2_p["mid_price"].round(0)

plt.scatter(dfSneg2["timestamp"], dfSneg2["price"], c = dfSneg2["quantity"], alpha = 0.6)
plt.colorbar(label='Quantity Traded')
#plt.scatter(dfSneg2_p["timestamp"], dfSneg2_p["mid_price"], color = "red")


"""
plt.scatter(dfAneg2["timestamp"], dfAneg2["price"], c = dfAneg2["quantity"])
plt.colorbar(label='Quantity Traded')
plt.scatter(dfAneg2_p["timestamp"], dfAneg2_p["mid_price"], color = "red")
"""
plt.show()
