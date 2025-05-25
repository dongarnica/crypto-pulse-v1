"""
Test script for Alpaca crypto trading client.
Demonstrates usage of the AlpacaCryptoTrading class with proper configuration management.
"""

import os
import sys

# Add the parent directory to the path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca_client import AlpacaCryptoTrading
from config.config import AppConfig

def main():
    """Main test function for Alpaca crypto trading."""
    
    # Initialize configuration
    config = AppConfig()
    
    # Get Alpaca configuration
    alpaca_config = config.get_alpaca_config()
    
    # Validate required keys are present
    if not alpaca_config['api_key'] or not alpaca_config['api_secret']:
        print("Error: Missing Alpaca API credentials!")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")
        return
    
    print("Initializing Alpaca client...")
    
    # Initialize the client with config
    client = AlpacaCryptoTrading(
        api_key=alpaca_config['api_key'],
        api_secret=alpaca_config['api_secret'],
        base_url=alpaca_config['base_url']
    )

    
    try:
        # Get account information
        print("Getting account information...")
        account = client.get_account()
        print(f"Account balance: ${account.get('cash', 'N/A')}")
        print(f"Buying power: ${account.get('buying_power', 'N/A')}")
        print(f"Portfolio value: ${account.get('portfolio_value', 'N/A')}")
        print(f"Day trade count: {account.get('daytrade_count', 'N/A')}")
        
        # Get current positions
        print("\nGetting current positions...")
        positions = client.list_positions()
        
        if positions and len(positions) > 0:
            print(f"Found {len(positions)} open position(s):")
            total_market_value = 0
            total_unrealized_pl = 0
            
            for i, pos in enumerate(positions, 1):
                symbol = pos.get('symbol', 'N/A')
                qty = float(pos.get('qty', 0))
                side = pos.get('side', 'N/A')
                market_value = float(pos.get('market_value', 0))
                avg_entry_price = float(pos.get('avg_entry_price', 0))
                current_price = float(pos.get('current_price', 0))
                unrealized_pl = float(pos.get('unrealized_pl', 0))
                unrealized_plpc = float(pos.get('unrealized_plpc', 0)) * 100
                
                print(f"\n  Position {i}: {symbol}")
                print(f"    Quantity: {qty} ({side})")
                print(f"    Avg Entry Price: ${avg_entry_price:.2f}")
                print(f"    Current Price: ${current_price:.2f}")
                print(f"    Market Value: ${market_value:.2f}")
                print(f"    Unrealized P&L: ${unrealized_pl:.2f} ({unrealized_plpc:+.2f}%)")
                
                total_market_value += market_value
                total_unrealized_pl += unrealized_pl
            
            print(f"\n  Portfolio Summary:")
            print(f"    Total Market Value: ${total_market_value:.2f}")
            print(f"    Total Unrealized P&L: ${total_unrealized_pl:.2f}")
        else:
            print("No open positions found")
        
        # Get current market price
        print("\nGetting current BTC price...")
        last_trade = client.get_last_trade("BTCUSD")
        current_price = float(last_trade['price'])
        print(f"Current BTC price: ${current_price:.2f}")
        
        # Example trade parameters (small amount for testing)
        test_qty = 0.001  # Very small amount for testing
        stop_loss_pct = 0.95  # 5% stop loss
        take_profit_pct = 1.05  # 5% take profit
        
        print(f"\nTesting trade with:")
        print(f"  Quantity: {test_qty} BTC")
        print(f"  Stop Loss: ${current_price * stop_loss_pct:.2f} ({stop_loss_pct*100-100:+.1f}%)")
        print(f"  Take Profit: ${current_price * take_profit_pct:.2f} ({take_profit_pct*100-100:+.1f}%)")
        
        # Place a test trade with stop loss and take profit
        print("\nPlacing test trade...")
        result = client.trade_with_limits(
            symbol="BTCUSD",
            qty=test_qty,
            side="buy",
            stop_loss=current_price * stop_loss_pct,
            take_profit=current_price * take_profit_pct
        )
        
        print("✅ Trade placed successfully!")
        print(f"Main order ID: {result['main_order'].get('id', 'N/A')}")
        print(f"Stop loss order ID: {result['stop_loss_order'].get('id', 'N/A')}")
        print(f"Take profit order ID: {result['take_profit_order'].get('id', 'N/A')}")
        
        # List current orders
        print("\nCurrent open orders:")
        orders = client.list_orders()
        if orders:
            for order in orders:
                print(f"  Order {order.get('id', 'N/A')}: {order.get('side', 'N/A')} {order.get('qty', 'N/A')} {order.get('symbol', 'N/A')} @ ${order.get('limit_price', 'N/A')}")
        else:
            print("  No open orders")
        
        # Uncomment the line below to cancel all orders (use with caution!)
        # print("\nCancelling all orders...")
        # client.cancel_all_orders()
        # print("✅ All orders cancelled")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("Please check your API credentials and network connection")

if __name__ == "__main__":
    main()