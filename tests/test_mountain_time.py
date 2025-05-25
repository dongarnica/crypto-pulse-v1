#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspaces/crypto-refactor')

import pytz
from datetime import datetime
from data.crypto_data_client import CryptoMarketDataClient

def test_mountain_time():
    print("Testing Mountain Time conversion...")
    
    # Test basic timezone conversion
    mountain_tz = pytz.timezone('America/Denver')
    utc_now = datetime.now(tz=pytz.UTC)
    mt_now = utc_now.astimezone(mountain_tz)
    
    print(f"Current UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Current Mountain Time: {mt_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Test the client's conversion method
    client = CryptoMarketDataClient()
    
    # Test with current timestamp
    current_ts = int(datetime.now().timestamp())
    mt_formatted = client._convert_to_mountain_time(current_ts)
    print(f"Client conversion: {mt_formatted}")
    
    # Test with None/invalid timestamp
    invalid_mt = client._convert_to_mountain_time(None)
    print(f"Invalid timestamp: {invalid_mt}")
    
    print("\nTesting API call...")
    try:
        # Test API call with a simple request
        btc_data = client.get_realtime_price("BTC/USD")
        if btc_data:
            print(f"BTC/USD Price: ${btc_data['price']:,.2f}")
            print(f"24h Change: {btc_data['24h_change']:+.2f}%")
            print(f"Last Updated (MT): {btc_data['last_updated_mt']}")
            print(f"Raw Timestamp: {btc_data['last_updated']}")
        else:
            print("Failed to get BTC data from API")
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    test_mountain_time()
