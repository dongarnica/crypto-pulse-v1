#!/usr/bin/env python3
"""
Minimal test to check basic functionality
"""

try:
    print("Testing basic imports...")
    
    # Test config import
    from config.config import AppConfig
    print("✅ Config import successful")
    
    # Test config creation
    config = AppConfig()
    print(f"✅ Config created for ticker: {config.ticker_short}")
    
    # Test alpaca client import
    from exchanges.alpaca_client import AlpacaCryptoTrading
    print("✅ Alpaca client import successful")
    
    # Test config access methods
    alpaca_config = config.get_alpaca_config()
    print(f"✅ Alpaca config retrieved: {list(alpaca_config.keys())}")
    
    print("\n=== Configuration Status ===")
    print(f"API Key set: {'Yes' if alpaca_config.get('api_key') else 'No'}")
    print(f"API Secret set: {'Yes' if alpaca_config.get('api_secret') else 'No'}")
    print(f"Base URL: {alpaca_config.get('base_url')}")
    
    if alpaca_config.get('api_key') and alpaca_config.get('api_secret'):
        print("\n✅ All credentials are configured!")
        print("Ready for live testing.")
    else:
        print("\n❌ Missing credentials")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET environment variables")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
