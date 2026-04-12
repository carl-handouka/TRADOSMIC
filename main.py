from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
# prendre en compte correl pour skew

class Trader:

    #✅ PARAMÈTRES OPTIMAUX HARDCODÉS
    #PARAMS = {"EMERALDS": {"alpha": 0.0, "depth_ticks": -0.22222222222222232, "tick_size": 8}, "TOMATOES": {"alpha": 0.3, "depth_ticks": -0.22222222222222232, "tick_size": 1}}
    PARAMS = {'EMERALDS': {'alpha': 0.08, 'depth_ticks': -0.6938775510204083, 'tick_size': 1}, 'TOMATOES': {'alpha': 0.1, 'depth_ticks': -0.6938775510204083, 'tick_size': 1}}
    #PARAMS = {'EMERALDS': {'alpha': 0.027054990312704974, 'depth_ticks': -0.5739503465234019, 'tick_size': 1}, 'TOMATOES': {'alpha': 0.12289900134843756, 'depth_ticks': -0.6713269937986717, 'tick_size': 1}}

    POSITION_LIMIT = {"EMERALDS": 20, "TOMATOES": 20}  # max position for each product
    ORDER_SIZE = {"EMERALDS": 200, "TOMATOES": 200}  # size of each order (can be adjusted based on the product and strategy)

    # def round_to_tick(self, value, tick):
    #     return int(round(value / tick) * tick)

    def compute_quotes(self, product, best_bid, best_ask, position):
        params = self.PARAMS[product]

        alpha = params["alpha"]
        depth_ticks = params["depth_ticks"]
        tick = params["tick_size"]

        # skew selon position
        shift = -alpha * position * tick

        bid = int(best_bid - depth_ticks * tick + shift)
        ask = int(best_ask + depth_ticks * tick + shift)

        # sécurité
        if bid >= ask:
            print(f"Warning: bid {bid} >= ask {ask}, adjusting to ensure bid < ask")
            mid = (best_bid + best_ask) // 2
            bid = min(bid, mid - tick)
            ask = max(ask, mid + tick)

        return bid, ask

    def run(self, state: TradingState):

        result = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            position = state.position.get(product, 0)

            bid, ask = self.compute_quotes(product, best_bid, best_ask, position)

            # gestion du risque (limite de position)
            buy_size = min(self.ORDER_SIZE[product], self.POSITION_LIMIT[product] - position)
            sell_size = min(self.ORDER_SIZE[product], self.POSITION_LIMIT[product] + position)

            if buy_size > 0:
                orders.append(Order(product, bid, buy_size))

            if sell_size > 0:
                orders.append(Order(product, ask, -sell_size))

            result[product] = orders

        return result, 0, "mm_position_strategy"

