import numpy as np
from datamodel import Order, TradingState

class Trader:
    
    def __init__(self, history_window=5):
        """
        Initialize the Trader with a rolling window for spread history.
        """
        self.history_spreads = []  # To track historical spreads
        self.history_window = history_window  # Size of the history window (number of past iterations)

    def calculate_mean_spread(self) -> float:
        """
        Calculate the mean of the spread over the history.
        """
        if len(self.history_spreads) == 0:
            return 0  # Default if there's no history
        return np.mean(self.history_spreads)

    def calculate_std_deviation(self) -> float:
        """
        Calculate the standard deviation of the spread over the history.
        """
        if len(self.history_spreads) < 2:
            return 0  # Standard deviation is not defined for less than 2 points
        return np.std(self.history_spreads)

    def run(self, state: TradingState):
        """
        The run method called at each iteration to decide on trading actions.
        """
        result = {}

        # Iterate over each product in the market
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            
            # Extract the best bid and ask prices for the product
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0] if len(order_depth.buy_orders) > 0 else (None, 0)
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0] if len(order_depth.sell_orders) > 0 else (None, 0)
            
            # If no bid or ask price is available, skip to the next product
            if not best_bid or not best_ask:
                continue
            
            # Calculate the current spread
            current_spread = best_ask - best_bid
            self.history_spreads.append(current_spread)
            
            # Limit the history size to the rolling window
            if len(self.history_spreads) > self.history_window:
                self.history_spreads.pop(0)
            
            # Calculate the mean and standard deviation of the spread
            mean_spread = self.calculate_mean_spread()
            std_deviation = self.calculate_std_deviation()
            
            # Define the Bollinger Bands (Upper and Lower Bands)
            upper_band = mean_spread + 2 * std_deviation
            lower_band = mean_spread - 2 * std_deviation
            
            # Determine trading actions based on mean reversion logic
            if current_spread > upper_band:
                # Short the product (expecting the spread to revert)
                result[product] = [Order(product, best_ask, -best_ask_amount)]  # Short at ask price
                
            elif current_spread < lower_band:
                # Go long (expecting the spread to revert)
                result[product] = [Order(product, best_ask, best_ask_amount)]  # Buy at ask price
        result = trader.run(trading_state)
        
        if result:  # Check if result is not empty or None
            orders, conversion, traderData = result
        else:
    # Handle the case where no data is returned
            print("No data returned from trader.run()")

        return result