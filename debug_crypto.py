#!/usr/bin/env python3

import sys
import time
print("Starting debug script...")

try:
    print("1. Testing basic imports...")
    import requests
    print("   - requests imported")
    
    import pandas as pd
    print("   - pandas imported")
    
    import numpy as np
    print("   - numpy imported")
    
    print("2. Testing API connectivity...")
    response = requests.get('https://api.coingecko.com/api/v3/ping', timeout=10)
    print(f"   - API ping status: {response.status_code}")
    print(f"   - API ping response: {response.text}")
    
    print("3. Testing crypto data client import...")
    from data.crypto_data_client import CryptoMarketDataClient
    print("   - CryptoMarketDataClient imported")
    
    print("4. Creating client...")
    client = CryptoMarketDataClient()
    print("   - Client created")
    
    print("5. Testing basic API call...")
    btc_data = client.get_realtime_price("BTC/USD")
    print(f"   - Real-time data: {btc_data}")
    
    print("6. Testing historical bars...")
    bars = client.get_historical_bars('BTC/USD', hours=24)
    print(f"   - Historical bars type: {type(bars)}")
    if hasattr(bars, 'shape'):
        print(f"   - Historical bars shape: {bars.shape}")
    
    print("Debug script completed successfully!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
