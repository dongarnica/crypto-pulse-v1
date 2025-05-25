#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspaces/crypto-refactor')

from data.crypto_data_client import CryptoMarketDataClient

def test_historical_data():
    print("Testing Historical Data Formatting...")
    
    client = CryptoMarketDataClient()
    
    # Test historical data for ETH/USD
    print("\nFetching ETH/USD 7-day historical data...")
    try:
        eth_history = client.get_historical_data("ETH/USD", days=7)
        if eth_history:
            print("Raw data keys:", eth_history.keys())
            print("Number of price points:", len(eth_history.get('prices', [])))
            
            # Use the new formatting method
            summary = client.format_historical_summary(eth_history, "ETH/USD")
            print("\nFormatted Summary:")
            print(summary)
        else:
            print("Failed to get historical data")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with BTC as well
    print("\n" + "="*50)
    print("Fetching BTC/USD 3-day historical data...")
    try:
        btc_history = client.get_historical_data("BTC/USD", days=3)
        if btc_history:
            summary = client.format_historical_summary(btc_history, "BTC/USD")
            print("\nFormatted Summary:")
            print(summary)
        else:
            print("Failed to get BTC historical data")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_historical_data()
