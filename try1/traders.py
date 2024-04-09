from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import pandas as pd

class Trader:

    def run(self, state: TradingState): # state must be a TradingState object
        print(state.traderData)
        print(f"Positions: {state.position}")
        
		# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths: # for each product in the order listing
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = self.calculate_fair_value(order_depth)  # TODO: Participant should calculate this value, FAIR VALUE PRICE (CREATE ALGORITHM)
            orderTriggered = False
            
            print("Acceptable price : " + str(self.calculate_fair_value(order_depth)))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            if len(order_depth.sell_orders) != 0: # if there are sell orders, you can buy
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]

                if int(best_ask) < acceptable_price: # SEND BUY ORDER
                    buy_order_amount = self.calculate_amount_to_buy(OrderDepth, state.position, product)
                    print("BUY", str(-buy_order_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -buy_order_amount)) # TODO: CAN CHANGE THE AMOUNT WE WANT TO BUY
                    orderTriggered = True
    
            if len(order_depth.buy_orders) != 0: # if there are buy orders, you can sell
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    sell_order_amount = self.calculate_amount_to_sell(OrderDepth, state.position, product)
                    print("SELL", str(-sell_order_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -sell_order_amount)) # TODO: CHANGE THIS TOO
                    orderTriggered = True
            
            if not orderTriggered:
                 pass

            result[product] = orders # add to result dictionary
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = f"Previous Acceptable: {acceptable_price}" # TODO: CHANGE THIS

				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData
    

    def calculate_fair_value(self, order_depth: OrderDepth):
        if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        else:
             best_ask = 0
        
        if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        else:
             best_bid = 0

        print(order_depth.buy_orders)

        sum = best_bid + best_ask
        avg = sum / 2
        
        return avg
    
    def calculate_amount_to_buy(self, order_depth: OrderDepth, position, product: str) -> int:
        limit: int

        if product == "AMETHYSTS":
             limit = 20
        elif product == "STARFRUIT":
             limit = 20

        if product in position:
            current_position = position[product]
        else: 
            current_position = 0
        
        print(f"current position {current_position}")
        if current_position < limit:
            return 1

        return 0

    def calculate_amount_to_sell(self, order_depth: OrderDepth, position, product: str) -> int:
        limit: int

        if product == "AMETHYSTS":
             limit = -20
        elif product == "STARFRUIT":
             limit = -20

        if product in position:
            current_position = position[product]
        else:
            current_position = 0

        print(f"current position {current_position}")
        if current_position > limit:
            return 1

        return 0

