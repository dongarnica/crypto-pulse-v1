#!/usr/bin/env python3
"""
Simple test to verify Alpaca positions functionality.
"""

import os
import sys
import traceback

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def main():
    print("=== Simple Alpaca Positions Test ===")
    
    try:
        # Test 1: Import config
        print("1. Testing config import...")
        from config.config import AppConfig
        config = AppConfig()
        print(f"   ✅ Config loaded for ticker: {config.ticker_short}")
        
        # Test 2: Check credentials
        print("2. Checking credentials...")
        alpaca_config = config.get_alpaca_config()
        api_key = alpaca_config.get('api_key', '')
        api_secret = alpaca_config.get('api_secret', '')
        
        print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")
        print(f"   API Secret: {'✅ Set' if api_secret else '❌ Missing'}")
        print(f"   Base URL: {alpaca_config.get('base_url', 'N/A')}")
        
        if not api_key or not api_secret:
            print("   ❌ Cannot proceed without credentials")
            return
        
        # Test 3: Import and initialize Alpaca client
        print("3. Testing Alpaca client import...")
        from exchanges.alpaca_client import AlpacaCryptoTrading
        
        client = AlpacaCryptoTrading(
            api_key=api_key,
            api_secret=api_secret,
            base_url=alpaca_config['base_url']
        )
        print("   ✅ Alpaca client initialized")
        
        # Test 4: Try to get account info
        print("4. Testing account access...")
        try:
            account = client.get_account()
            print(f"   ✅ Account accessed - Status: {account.get('status', 'Unknown')}")
            print(f"   Cash: ${account.get('cash', 'N/A')}")
            print(f"   Buying Power: ${account.get('buying_power', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️  Account access failed: {str(e)[:100]}...")
        
        # Test 5: Try to get positions
        print("5. Testing positions access...")
        try:
            positions = client.list_positions()
            print(f"   ✅ Positions retrieved - Count: {len(positions) if positions else 0}")
            
            if positions:
                for i, pos in enumerate(positions[:3], 1):  # Show first 3 positions
                    symbol = pos.get('symbol', 'N/A')
                    qty = pos.get('qty', 'N/A')
                    market_value = pos.get('market_value', 'N/A')
                    print(f"   Position {i}: {symbol} - {qty} units - ${market_value}")
            else:
                print("   📭 No positions found")
                
        except Exception as e:
            print(f"   ⚠️  Positions access failed: {str(e)[:100]}...")
        
        print("\n✅ Test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Check that all required modules are available")
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
