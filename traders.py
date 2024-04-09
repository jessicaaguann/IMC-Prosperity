from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState): # state must be a TradingState objet
        print("traderData: " + state.traderData) 
        print("Observations: " + str(state.observations))

		# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths: # for each product in the order listing
            order_depth: OrderDepth = state.order_depths[product] # save the order depth object into order_depth
            orders: List[Order] = [] # initialize order list (append orders for this specific product)
            acceptable_price = 10  # TODO: Participant should calculate this value, FAIR VALUE PRICE (CREATE ALGORITHM)
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            #                            The number of listed prices for buy orders in the order_depth object 
            #                                                                       number of listed prices for sell orders ^ 

            if len(order_depth.sell_orders) != 0: # if there are sell orders, you can buy
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                # gets the 0th element of the sell order, which must be the best asking price, and sets best_ask and best_ask_amount to
                # these values 
                if int(best_ask) < acceptable_price: # SEND BUY ORDER
                    print("BUY", str(-best_ask_amount) + "x", best_ask) # negative best_ask_amount because sell_orders are negative
                    orders.append(Order(product, best_ask, -best_ask_amount)) # TODO: CAN CHANGE THE AMOUNT WE WANT TO BUY
    
            if len(order_depth.buy_orders) != 0: # if there are buy orders, you can sell
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount)) # TODO: CHANGE THIS TOO
            
            result[product] = orders # add to result dictionary
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" # TODO: CHANGE THIS
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData