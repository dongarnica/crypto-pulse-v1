#!/usr/bin/env python3
"""
Quick verification script that writes results to a file.
This helps verify functionality when terminal output isn't working properly.
"""

import os
import sys
import traceback
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.getcwd())

def write_log(message):
    """Write message to verification log file."""
    with open('verification_log.txt', 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

def main():
    """Run verification tests and log results."""
    
    write_log("=== Alpaca Positions Verification Test ===")
    
    try:
        # Test 1: Config import and initialization
        write_log("Testing config import...")
        from config.config import AppConfig
        config = AppConfig()
        write_log(f"✅ Config loaded for ticker: {config.ticker_short}")
        
        # Test 2: Check credentials
        alpaca_config = config.get_alpaca_config()
        api_key_set = bool(alpaca_config.get('api_key'))
        api_secret_set = bool(alpaca_config.get('api_secret'))
        
        write_log(f"API Key configured: {api_key_set}")
        write_log(f"API Secret configured: {api_secret_set}")
        write_log(f"Base URL: {alpaca_config.get('base_url', 'N/A')}")
        
        if not api_key_set or not api_secret_set:
            write_log("❌ Cannot proceed without credentials")
            return
        
        # Test 3: Alpaca client import
        write_log("Testing Alpaca client import...")
        from exchanges.alpaca_client import AlpacaCryptoTrading
        
        client = AlpacaCryptoTrading(
            api_key=alpaca_config['api_key'],
            api_secret=alpaca_config['api_secret'],
            base_url=alpaca_config['base_url']
        )
        write_log("✅ Alpaca client initialized successfully")
        
        # Test 4: Check available methods
        write_log("Testing available methods...")
        methods_to_test = [
            'get_account',
            'list_positions', 
            'get_portfolio_summary',
            'analyze_position_risk'
        ]
        
        for method_name in methods_to_test:
            if hasattr(client, method_name):
                write_log(f"✅ Method {method_name} available")
            else:
                write_log(f"❌ Method {method_name} missing")
        
        # Test 5: Basic API connectivity (with timeout)
        write_log("Testing basic API connectivity...")
        try:
            account = client.get_account()
            status = account.get('status', 'Unknown')
            cash = account.get('cash', 'N/A')
            write_log(f"✅ Account accessed - Status: {status}, Cash: ${cash}")
        except Exception as e:
            write_log(f"⚠️ Account access failed: {str(e)[:100]}...")
        
        # Test 6: Positions retrieval
        write_log("Testing positions retrieval...")
        try:
            positions = client.list_positions()
            position_count = len(positions) if positions else 0
            write_log(f"✅ Positions retrieved - Count: {position_count}")
            
            if positions:
                for i, pos in enumerate(positions[:2], 1):  # Log first 2 positions
                    symbol = pos.get('symbol', 'N/A')
                    qty = pos.get('qty', 'N/A')
                    market_value = pos.get('market_value', 'N/A')
                    write_log(f"Position {i}: {symbol} - {qty} units - ${market_value}")
        except Exception as e:
            write_log(f"⚠️ Positions retrieval failed: {str(e)[:100]}...")
        
        # Test 7: Portfolio summary
        write_log("Testing portfolio summary...")
        try:
            portfolio = client.get_portfolio_summary()
            if 'error' not in portfolio:
                total_value = portfolio['account']['portfolio_value']
                position_count = portfolio['positions']['count']
                total_pl = portfolio['positions']['total_unrealized_pl']
                write_log(f"✅ Portfolio summary - Value: ${total_value}, Positions: {position_count}, P&L: ${total_pl}")
                
                if portfolio['performance']['best_performer']:
                    best = portfolio['performance']['best_performer']
                    write_log(f"Best performer: {best['symbol']} ({best['return_pct']:+.2f}%)")
            else:
                write_log(f"⚠️ Portfolio summary failed: {portfolio['error']}")
        except Exception as e:
            write_log(f"⚠️ Portfolio summary error: {str(e)[:100]}...")
        
        write_log("✅ Verification test completed successfully!")
        
    except ImportError as e:
        write_log(f"❌ Import error: {str(e)}")
        
    except Exception as e:
        write_log(f"❌ Unexpected error: {str(e)}")
        write_log(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Clear previous log
    if os.path.exists('verification_log.txt'):
        os.remove('verification_log.txt')
    
    main()
    
    # Print log contents to screen if possible
    try:
        with open('verification_log.txt', 'r') as f:
            print(f.read())
    except:
        pass
