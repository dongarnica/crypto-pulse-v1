from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderClass
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import os


from config.config import TradingConfig

class AlpacaCryptoTrader:
    def __init__(self, config: TradingConfig = None, paper=True):
        """
        Initialize the Alpaca Crypto Trader using TradingConfig.
        If config is None, a default TradingConfig is created.
        """
        self.config = config or TradingConfig()
        self.api_key = self.config.alpaca_api_key
        self.secret_key = self.config.alpaca_secret
        self.base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=paper)
        # Crypto data client does not require keys for market data
        self.crypto_client = CryptoHistoricalDataClient()

    def get_account(self):
        """Get account information."""
        return self.trading_client.get_account()

    def get_positions(self):
        """Get all open positions."""
        return self.trading_client.get_all_positions()

    def get_position(self, symbol):
        """Get a specific position by symbol."""
        for position in self.trading_client.get_all_positions():
            if position.symbol == symbol:
                return position
        return None

    def close_position(self, symbol):
        """Close a specific position by symbol."""
        return self.trading_client.close_position(symbol)

    def close_all_positions(self):
        """Close all open positions."""
        return self.trading_client.close_all_positions()

    def submit_market_order(self, symbol, qty, side=OrderSide.BUY):
        """Submit a market order."""
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        return self.trading_client.submit_order(order_data)

    def submit_limit_order(self, symbol, qty, limit_price, side=OrderSide.BUY):
        """Submit a limit order."""
        order_data = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            limit_price=limit_price,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        return self.trading_client.submit_order(order_data)

    def submit_stop_order(self, symbol, qty, stop_price, side=OrderSide.SELL):
        """Submit a stop (stop loss) order."""
        order_data = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            stop_price=stop_price,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        return self.trading_client.submit_order(order_data)

    def submit_take_profit_order(self, symbol, qty, limit_price, side=OrderSide.SELL):
        """Submit a take profit (limit) order."""
        return self.submit_limit_order(symbol, qty, limit_price, side)

