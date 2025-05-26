#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from exchanges.alpaca_client import AlpacaCryptoTrading
from config.config import AppConfig

def test_crypto_endpoints():
    """Test the fixed crypto endpoints"""
    print("=" * 50)
    print("Testing Fixed Alpaca Crypto Endpoints")
    print("=" * 50)
    
    # Initialize configuration
    config = AppConfig()
    alpaca_config = config.get_alpaca_config()
    
    # Initialize client
    client = AlpacaCryptoTrading(
        api_key=alpaca_config['api_key'],
        api_secret=alpaca_config['api_secret'],
        base_url=alpaca_config['base_url']
    )
    
    test_symbols = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'YFIUSD', 'AAVEUSD', 'DOGEUSD', 'LINKUSD']
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        
        # Test last trade
        try:
            trade = client.get_last_trade(symbol)
            if 'error' in trade:
                print(f"❌ Trade Error: {trade['error']} - {trade.get('message', '')}")
            else:
                print(f"✅ Last Trade: ${trade.get('price', 'N/A')} at {trade.get('timestamp', 'N/A')}")
        except Exception as e:
            print(f"❌ Trade Exception: {str(e)}")
        
        # Test last quote
        try:
            quote = client.get_last_quote(symbol)
            if 'error' in quote:
                print(f"❌ Quote Error: {quote['error']} - {quote.get('message', '')}")
            else:
                bid = quote.get('bid', 'N/A')
                ask = quote.get('ask', 'N/A')
                print(f"✅ Last Quote: Bid ${bid} / Ask ${ask}")
        except Exception as e:
            print(f"❌ Quote Exception: {str(e)}")

if __name__ == "__main__":
    test_crypto_endpoints()
