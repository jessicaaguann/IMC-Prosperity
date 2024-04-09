from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np

class Trader:
    
    def run(self, state: TradingState): # state must be a TradingState object
        # print("traderData: " + state.traderData) 

        if len(state.traderData) == 0:
            # first list is to save mid_price for AMETHYSTS
            # second list is to save mid_price for STARFRUITS 
            records = {"AMETHYSTS": [], "STARFRUIT": []}
        else:
            records = jsonpickle.decode(state.traderData)
            

        result = {}
        for product in state.order_depths: # for each product in the order listing
            order_depth: OrderDepth = state.order_depths[product] # save the order depth object into order_depth
            orders: List[Order] = [] 

            if product == "AMETHYSTS":
                if product in state.position:
                    position = state.position["AMETHYSTS"]
                else:
                    position = 0
                
                pos_limit = 20
                buy_dif = pos_limit-position

                mid_price = self.get_midprice(order_depth=order_depth) # gives order_depth for specific product
                records[product].append(mid_price)
                if state.timestamp < 500:
                    moving_average = 10000
                else:
                    moving_average = self.get_moving_average(records[product], 10)

                if len(records[product]) > 100: # to not store that much data inside record
                    records[product].pop(0) # pop the oldest

                buy_price = int(round(moving_average)) - 1

                print("BUY", str(buy_dif) + "x", buy_price)
                orders.append(Order(product, buy_price, buy_dif))

                neg_limit = -20
                sell_dif = neg_limit-position
    
                sell_price = int(round(moving_average)) + 1

                print("SELL", str(sell_dif) + "x", sell_price)
                orders.append(Order(product, sell_price, sell_dif))

            """elif product == "STARFRUIT":
                mid_price = self.get_midprice(order_depth=order_depth) # gives order_depth for specific product
                records[product].append(mid_price)
                moving_average = self.get_moving_average(records[product], 20)
                moving_std = self.get_std(records[product])
                scaler = 2
                upper_limit = moving_average + scaler * moving_std
                lower_limit = moving_average - scaler * moving_std

                print("Moving Average", moving_average)
                print("Upper Bound", upper_limit)
                print("Lower Bound", lower_limit)
                print("Mid Price", mid_price)

                if len(records[product]) > 100: # to not store that much data inside record
                    records[product].pop(0) # pop the oldest

                acceptable_price = 10  # TODO: Participant should calculate this value, FAIR VALUE PRICE (CREATE ALGORITHM)
                print("Acceptable price : " + str(acceptable_price))
                print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

                if len(order_depth.sell_orders) != 0: # if there are sell orders, you can buy
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                
                    if best_ask < lower_limit:
                        amount_to_buy = self.get_amount_to_buy(order_depth, state.position, product)
                        if amount_to_buy < 0: # make sure amount_to_buy is positive
                            amount_to_buy *= -1
                        
                        print("BUY", str(amount_to_buy) + "x", best_ask)
                        orders.append(Order(product, best_ask, amount_to_buy))
        
                if len(order_depth.buy_orders) != 0: # if there are buy orders, you can sell
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                    if best_bid > upper_limit:
                        amount_to_sell = self.get_amount_to_sell(order_depth, state.position, product)
                        if amount_to_sell > 0: # make sure amount_sell is negative
                            amount_to_sell *= -1
                        print("SELL", str(amount_to_sell) + "x", best_bid)
                        orders.append(Order(product, best_bid, amount_to_sell))"""
            
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
        if len(list_of_mid_prices) < num:
            return sum(list_of_mid_prices) / len(list_of_mid_prices)
        
        return sum(list_of_mid_prices[-num:]) / num
    
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
        

