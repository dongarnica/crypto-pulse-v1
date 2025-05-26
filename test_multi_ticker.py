#!/usr/bin/env python3
"""Quick test script for multi-ticker functionality."""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())

try:
    from config.config import AppConfig
    
    print("=== Testing Multi-Ticker Configuration ===")
    
    # Test 1: Default configuration
    print("\n1. Default configuration:")
    config = AppConfig(ticker='BTC')
    print(f"   Multi-ticker enabled: {config.multi_ticker_enabled}")
    print(f"   Max active tickers: {config.max_active_tickers}")
    print(f"   Active tickers: {config.get_active_tickers()}")
    
    # Test 2: Enable multi-ticker programmatically
    print("\n2. Multi-ticker enabled:")
    config.multi_ticker_enabled = True
    config.max_active_tickers = 3
    print(f"   Multi-ticker enabled: {config.multi_ticker_enabled}")
    print(f"   Max active tickers: {config.max_active_tickers}")
    print(f"   Active tickers: {config.get_active_tickers()}")
    
    # Test 3: Check available tickers
    print(f"\n3. Available tickers: {config.available_tickers}")
    
    print("\n✅ Multi-ticker configuration test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
