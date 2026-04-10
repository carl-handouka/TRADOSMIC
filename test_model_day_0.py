import pandas as pd
from datamodel import OrderDepth, TradingState
from base import Trader

# Load CSV
df = pd.read_csv("prices_round_0_day_-1.csv", sep=";")

trader = Trader()

for _, row in df.iterrows():
    
    product = row["product"]

    # Build order book
    order_depth = OrderDepth()
    order_depth.buy_orders = {
        row["bid_price_1"]: row["bid_volume_1"],
    }
    order_depth.sell_orders = {
        row["ask_price_1"]: row["ask_volume_1"],
    }

    state = TradingState(
        traderData="",
        timestamp=row["timestamp"],
        listings={},
        order_depths={product: order_depth},
        own_trades={},
        market_trades={},
        position={},
        observations={}
    )

    result = trader.run(state)

    print(result)