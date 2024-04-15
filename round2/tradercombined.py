

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
            records = {"AMETHYSTS": [], "STARFRUIT": [], "ORCHIDSPNL": [0]}
        else:
            records = jsonpickle.decode(state.traderData)
        
        conversions = 0
        result = {}
        for product in state.order_depths: # for each product in the order listing
            order_depth: OrderDepth = state.order_depths[product] # save the order depth object into order_depth
            orders, conversionstemp = self.get_orders_and_conversions(state, records, product)
            print("Buy Orders:", order_depth.buy_orders)
            print("Sell Orders:", order_depth.sell_orders)
            result[product] = orders # add to result dictionary

            if product == "ORCHIDS": # only update conversions for ORCHIDS
                conversions = conversionstemp
        
        print("TOTAL ORCHID PNL: ", records["ORCHIDSPNL"][0])
        traderData = jsonpickle.encode(records)
        return result, conversions, traderData
    
    def get_orders_and_conversions(self, state, records, product):
        # if product == "ORCHIDS":
        #    return self.orchids_algorithm(state, records)
        if product == "AMETHYSTS":
            return self.amethysts_algorithm(state, records)
        if product == "STARFRUIT":
            return self.starfruit_algorithm(state, records)
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

    def starfruit_linreg(self, starfruit_prices):
        if len(starfruit_prices) < 3:
            return None
        
        coefficients = [0.28814739, 0.32004992, 0.39149667]
        intercept = 1.5327896165945276
        coefficients = np.array(coefficients)
        last_three_prices = starfruit_prices[-3:]

        last_three_prices = np.array(last_three_prices)
        
        result = np.dot(last_three_prices, coefficients) + intercept

        return result
    
    def orchids_algorithm(self, state: TradingState, records):
        product = "ORCHIDS"
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = [] 

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        buy_amount = 0
        sell_amount = 0

        bid_price, ask_price, import_tariff, export_tariff, transport_fees = self.get_orchids(state)
                
        # to buy from the SOUTH --> ask_price + import_tariff + export_tariff
        south_buy_price = ask_price + import_tariff + transport_fees
        south_sell_price = bid_price - export_tariff - transport_fees

        print("SOUTH BUY:" + str(south_buy_price))
        print("SOUTH SELL:" + str(south_sell_price))
        print("LOCAL BUY:", best_ask)
        print("LOCAL SELL:", best_bid)

        # maximize pnl
        conversion1, conversion1pnl = 0, 0
        conversion2, conversion2pnl = 0, 0

        # if the best bid in the local market is greater than the south_buy_price, sell in the local market, and buy in the foreign
        if best_bid > south_buy_price:
            if abs(best_bid_amount) > 10:
                buy_amount = 10
            else:
                buy_amount = best_bid_amount
            
            print("CONVERT (BUY)", str(buy_amount) + "x", south_buy_price) # CONVERT FOREIGN
            conversion1 = buy_amount
            print("SELL", str(-buy_amount) + "x", best_bid) # SELL LOCAL
            orders.append(Order(product, best_bid, buy_amount))

            price_dif = best_bid - south_buy_price
            conversion1pnl = buy_amount * price_dif

        # if south_sell_price is greater than the local best asking price, buy in the local market and sell in the foreign market
        if south_sell_price > best_ask:
            if abs(best_ask_amount) > 10: 
                sell_amount = -10
            else:
                sell_amount = -best_ask_amount
            
            print("CONVERT (SELL)", str(sell_amount) + "x", south_sell_price) # SELL FOREIGN
            conversion2 = sell_amount
            print("BUY", str(-sell_amount) + "x", best_ask) # BUY LOCAL
            orders.append(Order(product, best_ask, -sell_amount))

            price_dif = south_sell_price - best_ask
            conversion2pnl = -sell_amount * price_dif

        if conversion2pnl > conversion1pnl:
            conversions = conversion2
            print(f"Orchid PNL: {conversion2pnl}")
            records["ORCHIDSPNL"][0] += conversion2pnl
        else:
            conversions = conversion1
            print(f"Orchid PNL: {conversion1pnl}")
            records["ORCHIDSPNL"][0] += conversion2pnl

        return orders, conversions

    def amethysts_algorithm(self, state: TradingState, records):
        product = "AMETHYSTS"
        order_depth: OrderDepth = state.order_depths[product] # save the order depth object into order_depth
        orders: List[Order] = [] 

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        buy_amount = 0
        sell_amount = 0
        conversions = 0
        
        position = self.get_position(product, state) 
                
        margin = 3
        buy_price = 10000 - margin
        sell_price = 10000 + margin
        
        if best_ask <= 9998:
            buy_amount = self.get_amount_to_buy(order_depth=order_depth, position=state.position, product=product)

            if buy_amount < 0:
                buy_amount *= -1
            
            print("BUY:", str(buy_amount) + "x", best_ask)
            orders.append(Order(product, best_ask, buy_amount))

        if best_bid >= 10002:
            sell_amount = self.get_amount_to_sell(order_depth=order_depth, position=state.position, product=product)

            if sell_amount > 0:
                sell_amount *= -1
            
            print("SELL:", str(sell_amount) + "x", best_bid)
            orders.append(Order(product, best_bid, sell_amount))
        
        pos_limit = 20 - buy_amount
        buy_dif = pos_limit-position

        mid_price = self.get_midprice(order_depth=order_depth) # gives order_depth for specific product
        records[product].append(mid_price)

        if len(records[product]) > 100: # to not store that much data inside record
            records[product].pop(0) # pop the oldest

        print("BUY", str(buy_dif) + "x", buy_price)
        orders.append(Order(product, buy_price, buy_dif))

        neg_limit = -20 - sell_amount
        sell_dif = neg_limit-position

        print("SELL", str(sell_dif) + "x", sell_price)
        orders.append(Order(product, sell_price, sell_dif))

        return orders, conversions
    
    def starfruit_algorithm(self, state: TradingState, records):
        product = "STARFRUIT"
        order_depth: OrderDepth = state.order_depths[product] # save the order depth object into order_depth
        orders: List[Order] = [] 
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        buy_amount = 0
        sell_amount = 0
        mid_price = self.get_midprice(order_depth)
        position = self.get_position(product, state)
        
        records['STARFRUIT'].append(mid_price)
        fair_value = self.starfruit_linreg(records['STARFRUIT'])

        print(f"Fair Value: {fair_value}")
        if len(records['STARFRUIT']) > 5:
            records['STARFRUIT'].pop(0)
        
        if fair_value is None:
            pass
        else:
            if best_ask < fair_value: # if the best ask price is less than the fair_value
                buy_amount = abs(self.get_amount_to_buy(order_depth, state.position, product))

                print("BUY", str(buy_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, buy_amount))

            if best_bid > fair_value: # if the best bid price is greater than the fair_value price
                sell_amount = -abs(self.get_amount_to_sell(order_depth, state.position, product))

                print("SELL", str(sell_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, sell_amount))

            margin = 2
            fair_value = int(round(fair_value))
            buy_price = fair_value - margin
            sell_price = fair_value + margin

            pos_limit = 20 - buy_amount
            buy_dif = pos_limit-position

            print("BUY", str(buy_dif) + "x", buy_price)
            orders.append(Order(product, buy_price, buy_dif))

            neg_limit = -20 - sell_amount
            sell_dif = neg_limit-position

            print("SELL", str(sell_dif) + "x", sell_price)
            orders.append(Order(product, sell_price, sell_dif))

        return orders, 0