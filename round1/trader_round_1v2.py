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
            records = {"AMETHYSTS": [], "STARFRUIT": [], "STARSMA": []}
        else:
            records = jsonpickle.decode(state.traderData)
            
        result = {}
        for product in state.order_depths: # for each product in the order listing
            order_depth: OrderDepth = state.order_depths[product] # save the order depth object into order_depth
            orders: List[Order] = [] 
            print("Buy Orders:", order_depth.buy_orders)
            print("Sell Orders:", order_depth.sell_orders)

            if product == "AMETHYSTS":
                if product in state.position:
                    position = state.position["AMETHYSTS"]
                else:
                    position = 0
                
                pos_limit = 20
                buy_dif = pos_limit-position

                mid_price = self.get_midprice(order_depth=order_depth) # gives order_depth for specific product
                records[product].append(mid_price)

                if len(records[product]) > 100: # to not store that much data inside record
                    records[product].pop(0) # pop the oldest

                buy_price = 10000 - 2

                print("BUY", str(buy_dif) + "x", buy_price)
                orders.append(Order(product, buy_price, buy_dif))

                neg_limit = -20
                sell_dif = neg_limit-position
    
                sell_price = 10000 + 2

                print("SELL", str(sell_dif) + "x", sell_price)
                orders.append(Order(product, sell_price, sell_dif))

            if product == "STARFRUIT":
                mid_price = self.get_midprice(order_depth=order_depth) # gives order_depth for specific product
                records[product].append(mid_price)

                small_window_size = 25
                large_window_size = 80
                moving_average = self.get_moving_average(records[product], small_window_size)
                ema = self.get_ema(records[product], large_window_size)
                moving_std = self.get_std(records[product])
                scaler = 1.82
                upper_limit = moving_average + scaler * moving_std
                lower_limit = moving_average - scaler * moving_std

                print("Moving Average", moving_average)
                print("EMA", ema)
                print("Upper Bound", upper_limit)
                print("Lower Bound", lower_limit)
                print("Mid Price", mid_price)

                if len(records[product]) > 100: # to not store that much data inside record
                    records[product].pop(0) # pop the oldest

                print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                if(best_ask < lower_limit): # lower band
                    buy_amount = self.get_amount_to_buy(order_depth=order_depth, position=state.position, product=product)

                    if buy_amount < 0:
                        buy_amount *= -1
                    
                    print("BUY:", str(buy_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, buy_amount))
                
                if(best_bid > upper_limit): # upper band
                    sell_amount = self.get_amount_to_sell(order_depth=order_depth, position=state.position, product=product)

                    if sell_amount > 0:
                        sell_amount *= -1
                    
                    print("SELL:", str(sell_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, sell_amount))
                
            result[product] = orders # add to result dictionary
        
        traderData = jsonpickle.encode(records) # TODO: CHANGE THIS
        
        conversions = 1
        return result, conversions, traderData
    
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
        list_to_series = pd.Series(list_of_mid_prices)
        return list_to_series.ewm(span=period, adjust=False).mean().iloc[-1]

    def get_std(self, list_of_mid_prices: list) -> float:
        return np.std(list_of_mid_prices)
    
    def get_amount_to_buy(self, order_depth: OrderDepth, position, product: str) -> int:
        limit: int

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]

        if product == "AMETHYSTS":
             limit = 20
        elif product == "STARFRUIT":
             limit = 20

        if product in position:
            current_position = position[product]
        else: 
            current_position = 0

        print("Cur pos: ", current_position, "; Ask Amount: ", best_ask_amount)
        if current_position - best_ask_amount <= limit:
            print(f"Buy Amount: {-best_ask_amount}")
            return -best_ask_amount
        else:
            print(f"Buy Amount: {limit-current_position}")
            return limit-current_position
        

    def get_amount_to_sell(self, order_depth: OrderDepth, position, product: str) -> int:
        limit: int

        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        if product == "AMETHYSTS":
             limit = -20
        elif product == "STARFRUIT":
             limit = -20

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
        

