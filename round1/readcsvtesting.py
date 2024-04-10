import pandas as pd
import matplotlib.pyplot as plt 

df0_t = pd.read_csv("./IMC testing/round1/trades_round_1_day_-1_nn.csv", delimiter=';') 
df0_p = pd.read_csv("./IMC testing/round1/prices_round_1_day_-1.csv", delimiter=';')

def split_df_trades(df):
    dfA = df[df["symbol"] == "AMETHYSTS"]
    dfS = df[df["symbol"] == "STARFRUIT"]
    return dfA, dfS

def split_df_prices(df):
    dfA = df[df["product"] == "AMETHYSTS"]
    dfS = df[df["product"] == "STARFRUIT"]
    return dfA, dfS

df0_pa, df0_ps = split_df_prices(df0_p)
df0_ta, df0_ts = split_df_trades(df0_t)


# FILTERING
df0_ts = df0_ts[df0_ts["timestamp"] < 100000]
df0_ps = df0_ps[df0_ps["timestamp"] < 100000]

# Moving averages
df0_ps_ma = df0_ps["mid_price"].rolling(window = 20).mean()
df0_ps_ma_long = df0_ps["mid_price"].rolling(window = 50).mean()

# PLOTTING
plt.scatter(df0_ps["timestamp"], df0_ps["bid_price_1"], color = "green", s = 10, alpha = 0.6)
plt.scatter(df0_ps["timestamp"], df0_ps["bid_price_2"], color = "green", s = 10, alpha = 0.6)
plt.scatter(df0_ps["timestamp"], df0_ps["bid_price_3"], color = "green", s = 10, alpha = 0.6)

plt.scatter(df0_ps["timestamp"], df0_ps["ask_price_1"], color = "red", s = 10, alpha = 0.6)
plt.scatter(df0_ps["timestamp"], df0_ps["ask_price_2"], color = "red", s = 10, alpha = 0.6)
plt.scatter(df0_ps["timestamp"], df0_ps["ask_price_3"], color = "red",s = 10, alpha = 0.6)

# plt.scatter(df0_ts["timestamp"], df0_ts["price"], color = "blue", s = 20, alpha = 0.8)

plt.plot(df0_ps["timestamp"], df0_ps_ma, label = "Short SMA")
plt.plot(df0_ps["timestamp"], df0_ps_ma_long, label = "Long SMA")

plt.legend()
plt.show()
