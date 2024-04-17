import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Read the CSV file into a DataFrame
df = pd.read_csv('./IMC testing/round3/log5.csv', delimiter=';')

# only keep rows with products as gift basket, chocolate, roses, strawberries
df = df[df['product'].isin(['GIFT_BASKET', 'CHOCOLATE', 'ROSES', 'STRAWBERRIES'])]

# get gift basket rows
df_gb = df[df['product'] == 'GIFT_BASKET'].reset_index(drop = True)

# filter out rows
df.drop(df.columns.difference(['timestamp', 'mid_price', 'product']), axis = 1, inplace = True)
df_gb.drop(['timestamp', 'profit_and_loss', 'product', 'day'], axis = True, inplace = True)

# get the mid price for each product
gb_series = df[df['product'] == 'GIFT_BASKET']['mid_price'].reset_index(drop = True)
c_series = df[df['product'] == 'CHOCOLATE']['mid_price'].reset_index(drop = True)
r_series = df[df['product'] == 'ROSES']['mid_price'].reset_index(drop = True)
s_series = df[df['product'] == 'STRAWBERRIES']['mid_price'].reset_index(drop = True)

# gb = 4 * chocolate + 6 * strawberries + 1 * roses
sum_series = c_series * 4 + r_series + s_series * 6

# create a data frame that has the sum and the real price of gift basket
df_merged = pd.DataFrame()
df_merged['GIFT_BASKET'] = gb_series
df_merged['SUMMED_COMP'] = sum_series
df_merged['DIF'] = gb_series - sum_series
df_merged['timestamp'] = df.loc[df['product'] == 'STRAWBERRIES', 'timestamp'].reset_index(drop=True)

# print('Max dif', df_merged['DIF'].max())
# print('Min dif', df_merged['DIF'].min())

# get buy sell signals, depending on a and b
def get_signals(df_merged, a, b):
    signals_series = pd.Series('', index = df_merged.index, dtype = 'object')

    sell_condition = df_merged['DIF'] > a
    buy_condition = df_merged['DIF'] < b
    neither_condition = (df_merged['DIF'] < a) & (df_merged['DIF'] > b)

    signals_series[sell_condition] = 'S'
    signals_series[buy_condition] = 'B'
    signals_series[neither_condition] = 'N'

    return signals_series


# define limit for buying and selling gifts basket
limit = 60

# input a b and c, calculate PNL, optimize PNL
def calculate_PNL(df_merged, df, a, b, c):
    signals = get_signals(df_merged, a, b)
    df_gb = df.copy()

    df_gb['signal'] = signals
    df_gb['PNL'] = 0.0
    # df_gb['Losses'] = 0.0
    # df_gb['Gains'] = 0.0

    df_gb['buy_price'] = df_gb['mid_price'] + c
    df_gb['sell_price'] = df_gb['mid_price'] - c

    current_pos = 0
    min_pos_count = 0
    max_pos_count = 0

    for index, row in df_gb.iterrows():
        if row['signal'] == 'B':
            # Buy logic
            # if specific row has a buy_condition and buy_price is greater than ask_price_1, buy at ask_price_1 * ask_volume_1 * -1, put this product in PNL
            if row['buy_price'] > row['ask_price_1']:

                if current_pos <= limit:
                    buy_volume = row['ask_volume_1'] * -1

                    if (-buy_volume + current_pos) > limit:
                        buy_volume = -(limit - current_pos)

                    pnl = row['ask_price_1'] * buy_volume
                    df_gb.at[index, 'PNL'] = pnl
                    # df_gb.at[index, 'Losses'] = pnl
                    current_pos -= buy_volume
                else:
                    max_pos_count += 1

        elif row['signal'] == 'S':
            # Sell logic
            # if specific row has a sell_condition and sell_price is less than bid_price_1, sell at bid_price_1 * bid_volume_1, put this product in PNL
            if row['sell_price'] < row['bid_price_1']:
                if current_pos >= -limit:
                    sell_volume = row['bid_volume_1']

                    if (current_pos - sell_volume) < -limit:
                        sell_volume = abs(limit - current_pos)

                    pnl = row['bid_price_1'] * sell_volume
                    df_gb.at[index, 'PNL'] = pnl
                    # df_gb.at[index, 'Gains'] = pnl
                    current_pos -= sell_volume
                else:
                    min_pos_count += 1


    total_pnl = df_gb['PNL'].sum()
    # total_gains = df_gb['Gains'].sum()
    # total_losses = df_gb['Losses'].sum()
    
    """print(f"a: {a}, b: {b}, c: {c}")
    print("Total PNL:", total_pnl)
    print("Total Gains:", total_gains)
    print("Total Losses:", total_losses)
    print("Min position count", min_pos_count)
    print("Max position count", max_pos_count)"""
    
    return total_pnl


# init conditions
 
a = 480
b = 310
c = 200

# brute force find the max total_pnl
max_a, max_b, max_c = 0, 0, 0
max_pnl = 0

# Define initial conditions
a = 470
b = 315
c = 60


print(calculate_PNL(df_merged, df_gb, 430, 350, 3))

while a >= 300 and a > b: 
    print("A", a)
    while b <= 450 and b < a:
        print("B", b)
        c = 60
        while c >= 2:
            cur_pnl = calculate_PNL(df_merged, df_gb, a, b, c)
            if cur_pnl > max_pnl:
                print("Updated PNL", cur_pnl, f"A: {a}, B: {b}, C: {c}")
                max_pnl = cur_pnl
                max_a = a
                max_b = b
                max_c = c
            c -= 1
        b += 1
    a -= 1
    b = 315

print("Max PNL", max_pnl)
print(f"A: {max_a}, B: {max_b}, C: {max_c}")