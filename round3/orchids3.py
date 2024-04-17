from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import pandas as pd
import numpy as np

class Trader:
    
    def run(self, state: TradingState): # state must be a TradingState object
        # print("traderData: " + state.traderData) 

        if len(state.traderData) == 0:
            # first list is to save mid_price for AMETHYSTS
            # second list is to save mid_price for STARFRUITS 
            # ORCHIDSLMID = local mid price for orchids
            # ORCHIDSFMID = foreign/south mid price for orchids
            records = {"AMETHYSTS": [], "STARFRUIT": []}
        else:
            records = jsonpickle.decode(state.traderData)
        
        conversions = 0
        result = {}
        for product in state.order_depths: # for each product in the order listing
            order_depth: OrderDepth = state.order_depths[product] # save the order depth object into order_depth
            orders, conversions1 = self.get_orders_and_conversions(state= state, records= records, product=product)
            print("Buy Orders:", order_depth.buy_orders)
            print("Sell Orders:", order_depth.sell_orders)
            result[product] = orders # add to result dictionary

            if product == "ORCHIDS":
                conversions = conversions1

        traderData = jsonpickle.encode(records)
        print("Conversions", str(conversions))
        return result, conversions, traderData
    
    def get_orders_and_conversions(self, state, records, product):
        if product == "ORCHIDS":
            orders, conversions = self.orchids_algorithm(state, records)
            return orders, conversions
        
        return [], 0
        

    def get_midprice(self, order_depth : OrderDepth) -> float:
        count = 0
        if len(order_depth.sell_orders) != 0:
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            count += 1
        else:
            best_ask = 0

        if len(order_depth.buy_orders) != 0:
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            count+= 1
        else: 
            best_bid = 0

        if count == 0:
            return None
        
        return (best_ask + best_bid) / count
    
    def get_moving_average(self, list_of_mid_prices : list, num : int) -> float: 
        if len(list_of_mid_prices) == 0:
            return None
        
        if len(list_of_mid_prices) < num:
            return sum(list_of_mid_prices) / len(list_of_mid_prices)
        
        return sum(list_of_mid_prices[-num:]) / num
    
    def get_ema(self, list_of_mid_prices: list, period : int) -> float:
        period = min(period, len(list_of_mid_prices))

        if period == 0:
            return None
        
        list_to_series = pd.Series(list_of_mid_prices)
        return list_to_series.ewm(span=period, adjust=False).mean().iloc[-1]

    def get_std(self, list_of_mid_prices: list, window_size) -> float:
        if len(list_of_mid_prices) < window_size:
            window_size = len(list_of_mid_prices)
        
        return np.std(list_of_mid_prices[-window_size:])
    
    def get_amount_to_buy(self, order_depth: OrderDepth, position, product: str) -> int:
        limit: int

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]

        if product == "AMETHYSTS":
             limit = 20
        elif product == "STARFRUIT":
             limit = 20
        elif product == "ORCHIDS":
             limit = 100

        if product in position:
            current_position = position[product]
        else: 
            current_position = 0

        print("Cur pos: ", current_position, "; Ask Amount: ", best_ask_amount)
        if current_position - best_ask_amount <= limit:
            print(f"Buy Amount: {abs(best_ask_amount)}")
            return abs(best_ask_amount)
        else:
            print(f"Buy Amount: {limit-current_position}")
            return abs(limit-current_position)

    def get_amount_to_sell(self, order_depth: OrderDepth, position, product: str) -> int:
        limit: int

        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        if product == "AMETHYSTS":
             limit = -20
        elif product == "STARFRUIT":
             limit = -20
        elif product == "ORCHIDS":
            limit = -100

        if product in position:
            current_position = position[product]
        else:
            current_position = 0

        print(f"Cur pos: {current_position}; Bid Amount: {best_bid_amount}")
        if current_position - best_bid_amount >= limit:
            print(f"Sell Amount: {best_bid_amount}")
            return -best_bid_amount
        else:
            print(f"Sell Amount: {current_position-limit}")
            return current_position-limit
        
    def get_slope_of_ma(self, list, window_size, num_in_avg):
        if len(list) < window_size + num_in_avg:
            return None
        
        list_of_ma = []
        local_list = list.copy()
        for x in range(num_in_avg):
            ma = self.get_moving_average(local_list, window_size)
            local_list.pop()
            list_of_ma.append(ma)

        x = range(num_in_avg)
        x_mean = np.mean(x)
        ma_mean = np.mean(list_of_ma)

        numerator = np.sum((x - x_mean) * (ma_mean - list_of_ma))
        denominator = np.sum((x - x_mean) ** 2)
        slope = numerator / denominator

        return slope

    def predict_with_linear_regression(self, list_of_prices, window_size):
        if len(list_of_prices) == 0 or len(list_of_prices) == 1:
            return None
        
        if len(list_of_prices) < window_size:
            window_size = len(list_of_prices)

        recent_listings = list_of_prices[-window_size:] # getting the most recent window_size listings in the list

        X = np.arange(0, window_size).reshape(-1, 1) # time
        y = np.array(recent_listings) # values

        # adding bias
        X_with_bias = np.c_[X, np.ones(X.shape[0])] # concatenates a column of ones to the X to add a bias

        # Calculate coefficients using the normal equation (linear algebra --> A^TAx = A^Tb)
        coefficients = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)

        # use coefficients to calculate the next value, dot product between the [next_time, 1] and coefficients [m, b]
        next_time = window_size + 1
        next_price = np.dot(np.array([next_time, 1]), coefficients)

        return next_price

    def get_position(self, product, state):
        if product in state.position:
            position = state.position[product]
        else:
            position = 0
            
        return position

    def get_orchids(self, state: TradingState):
        bid_price = state.observations.conversionObservations['ORCHIDS'].bidPrice
        ask_price = state.observations.conversionObservations['ORCHIDS'].askPrice
        import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
        export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
        transport_fees = state.observations.conversionObservations['ORCHIDS'].transportFees

        return bid_price, ask_price, import_tariff, export_tariff, transport_fees

    def orchids_algorithm(self, state: TradingState, records):
        product = "ORCHIDS"
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = [] 

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        buy_amount = 0
        sell_amount = 0
        conversions = 0

        bid_price, ask_price, import_tariff, export_tariff, transport_fees = self.get_orchids(state)

        # to buy from the SOUTH --> ask_price + import_tariff + export_tariff
        south_buy_price = ask_price + import_tariff + transport_fees
        south_sell_price = bid_price - export_tariff - transport_fees

        # if high tariffs, do sf alg on orchids
        if south_sell_price > best_ask:
            buy_amount = self.get_amount_to_buy(order_depth, state.position, product)
            print(f"BUY LOCAL {buy_amount}x{best_ask}")
            orders.append(Order(product, best_ask, buy_amount))
            conversions = -buy_amount

        if south_buy_price < best_bid:
            sell_amount = self.get_amount_to_sell(order_depth, state.position, product)
            print(f"SELL LOCAL {sell_amount}x{best_bid}")
            orders.append(Order(product, best_bid, sell_amount))
            conversions = -sell_amount
    
        return orders, conversions
