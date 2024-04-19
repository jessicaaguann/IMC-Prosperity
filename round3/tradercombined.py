

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
            records = {"AMETHYSTS": [], "STARFRUIT": []}
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
        
        traderData = jsonpickle.encode(records)
        return result, conversions, traderData
    
    def get_orders_and_conversions(self, state, records, product):
        if product == "ORCHIDS":
           return self.orchids_algorithm(state, records)
        if product == "AMETHYSTS":
            return self.amethysts_algorithm(state, records)
        if product == "STARFRUIT":
            return self.starfruit_algorithm(state, records)
        if product == 'GIFT_BASKET':
            return self.gift_basket_algorithm(state, records)
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

    def get_std(self, list_of_mid_prices: list) -> float:
        return np.std(list_of_mid_prices)
    
    def get_amount_to_buy(self, order_depth: OrderDepth, position, product: str) -> int:
        limit: int = 0

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]

        if product == "AMETHYSTS":
            limit = 20
        elif product == "STARFRUIT":
            limit = 20
        elif product == "ORCHIDS":
            limit = 100
        elif product == 'GIFT_BASKET':
            limit = 60

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
        limit: int = 0

        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

        if product == "AMETHYSTS":
            limit = -20
        elif product == "STARFRUIT":
            limit = -20
        elif product == "ORCHIDS":
            limit = -100
        elif product == 'GIFT_BASKET':
            limit = -60

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
        
        # if the conversion buy price is slightly greater than what you can sell it at, buy now and try selling a bit higher
        elif south_buy_price - 0.5 < best_bid:
            sell_amount = self.get_amount_to_sell(order_depth, state.position, product)
            print(f"SELL LOCAL {sell_amount}x{best_bid}")
            orders.append(Order(product, best_bid + 1, sell_amount))
            conversions = -sell_amount
        elif south_buy_price - 2.5 < best_bid:
            sell_amount = self.get_amount_to_sell(order_depth, state.position, product)
            print(f"SELL LOCAL {sell_amount}x{best_bid}")
            orders.append(Order(product, best_bid + 3, sell_amount))
            conversions = -sell_amount
        elif south_buy_price - 3.5 < best_bid:
            sell_amount = self.get_amount_to_sell(order_depth, state.position, product)
            print(f"SELL LOCAL {sell_amount}x{best_bid}")
            orders.append(Order(product, best_bid + 4, sell_amount))
            conversions = -sell_amount
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
        
        records[product].append(mid_price)
        fair_value = self.starfruit_linreg(records[product])

        print(f"Fair Value: {fair_value}")
        if len(records[product]) > 5:
            records[product].pop(0)
        
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
    
    def gift_basket_components_total(self, state: TradingState):
        product = 'STRAWBERRIES'
        order_depth = state.order_depths[product]
        s_mid_price = self.get_midprice(order_depth=order_depth)

        product = 'CHOCOLATE'
        order_depth = state.order_depths[product]
        c_mid_price = self.get_midprice(order_depth = order_depth)

        product = 'ROSES'
        order_depth = state.order_depths[product]
        r_mid_price = self.get_midprice(order_depth = order_depth)

        # basket is made of 4 chocolate + 6 strawberries + 1 rose
        total = 6 * s_mid_price + r_mid_price + 4 * c_mid_price
        return total

    def gift_basket_algorithm(self, state, records):
        product = "GIFT_BASKET"

        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = [] 

        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        mid_price = self.get_midprice(order_depth) # change this to fair value price?

        buy_amount = 0
        sell_amount = 0
        conversions = 0

        total_components = self.gift_basket_components_total(state)

        dif = mid_price - total_components

        # TODO: FIND A, B, C
        a = 433 # positive difference limit
        b = 350 # negative difference limit
        c = 7 # margin to sell/buy

        sell_signal = False
        buy_signal = False

        if dif > a:
            sell_signal = True
        elif dif < b:
            buy_signal = True

        if sell_signal:
            sell_price : int = round(mid_price - c)
            sell_amount = -10

            if sell_price < best_bid:
                sell_amount = self.get_amount_to_sell(order_depth, state.position, product)
                print(f"SELL {sell_amount}x{sell_price}")
                orders.append(Order(product, sell_price, sell_amount))

        
        elif buy_signal:
            buy_price : int = round(mid_price + c)
            buy_amount = 10

            if buy_price > best_ask:
                buy_amount = self.get_amount_to_buy(order_depth, state.position, product)
                print(f"BUY {buy_amount}x{buy_price}")
                orders.append(Order(product, buy_price, buy_amount))
    
        return orders, conversions