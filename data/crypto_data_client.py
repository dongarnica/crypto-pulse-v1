import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio
import websockets
import json
import requests
from typing import Dict, Optional, Union
import pytz

class CryptoMarketDataClient:
    """
    Provides historical and real-time crypto market data using CoinGecko API.
    Supports symbol notation like BTC/USD.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        load_dotenv()
        self.mountain_tz = pytz.timezone('America/Denver')  # Mountain Time
        
    def _convert_to_mountain_time(self, timestamp: int) -> str:
        """
        Convert UTC timestamp to Mountain Time formatted string.
        
        Args:
            timestamp: Unix timestamp in seconds
            
        Returns:
            str: Formatted datetime string in Mountain Time
        """
        if not timestamp:
            return "N/A"
            
        # Convert from Unix timestamp to UTC datetime
        utc_dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        
        # Convert to Mountain Time
        mountain_dt = utc_dt.astimezone(self.mountain_tz)
        
        # Format as readable string
        return mountain_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        
    def _normalize_symbol(self, symbol: str) -> tuple:
        """
        Convert symbol format (e.g., BTC/USD) to CoinGecko format (bitcoin, usd)
        
        Args:
            symbol: Trading pair in format like BTC/USD
            
        Returns:
            tuple: (coin_id, vs_currency)
        """
        base, quote = symbol.split('/')
        coin_id = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'USDT': 'tether',
            'USD': 'usd',
            'USDC': 'usd-coin',
            'AAVE': 'aave',
            'BCH': 'bitcoin-cash',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'LTC': 'litecoin',
            'SUSHI': 'sushi',
            'UNI': 'uniswap',
            'XRP': 'ripple',
            'YFI': 'yearn-finance',
            # Add more mappings as needed
        }.get(base.upper(), base.lower())
        
        vs_currency = quote.lower()
        return coin_id, vs_currency

    def get_realtime_price(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time price data for a cryptocurrency pair.
        
        Args:
            symbol: Trading pair in format like BTC/USD
            
        Returns:
            dict: Market data for the requested cryptocurrency pair
        """
        coin_id, vs_currency = self._normalize_symbol(symbol)
        
        endpoint = "/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': vs_currency,
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        try:
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Get the raw timestamp
            last_updated_timestamp = data[coin_id].get('last_updated_at')
            
            return {
                'symbol': symbol,
                'price': data[coin_id][vs_currency],
                'market_cap': data[coin_id].get(f'{vs_currency}_market_cap'),
                '24h_vol': data[coin_id].get(f'{vs_currency}_24h_vol'),
                '24h_change': data[coin_id].get(f'{vs_currency}_24h_change'),
                'last_updated': last_updated_timestamp,
                'last_updated_mt': self._convert_to_mountain_time(last_updated_timestamp)
            }
        except Exception as e:
            print(f"Error fetching real-time data for {symbol}: {e}")
            return None

    async def get_realtime_websocket(self, symbols: list, callback: callable):
        """
        Connect to CoinGecko WebSocket for real-time updates (simulated with polling)
        
        Args:
            symbols: List of trading pairs in format like ['BTC/USD', 'ETH/USD']
            callback: Function to call when new data arrives
        """
        # Note: CoinGecko doesn't have a public WebSocket API, so we simulate with polling
        while True:
            for symbol in symbols:
                data = self.get_realtime_price(symbol)
                if data:
                    await callback(data)
            await asyncio.sleep(60)  # Poll every 60 seconds

    def get_historical_data(self, symbol: str, days: int = 1) -> Optional[Dict]:
        """
        Get historical data for a cryptocurrency pair.
        
        Args:
            symbol: Trading pair in format like BTC/USD
            days: Number of days of history to retrieve
            
        Returns:
            dict: Historical market data
        """
        coin_id, vs_currency = self._normalize_symbol(symbol)
        
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        
        try:
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None

    def format_historical_summary(self, historical_data: Dict, symbol: str) -> str:
        """
        Format historical data into a readable summary.
        
        Args:
            historical_data: Raw historical data from CoinGecko API
            symbol: Trading symbol for display
            
        Returns:
            str: Formatted summary of historical data
        """
        if not historical_data or 'prices' not in historical_data:
            return f"No historical data available for {symbol}"
        
        prices = historical_data['prices']
        if not prices:
            return f"No price data available for {symbol}"
        
        # Get first and last prices
        first_price = prices[0][1]  # [timestamp, price]
        last_price = prices[-1][1]
        
        # Calculate price change
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100
        
        # Get highest and lowest prices
        all_prices = [price[1] for price in prices]
        highest = max(all_prices)
        lowest = min(all_prices)
        
        # Convert timestamps to Mountain Time
        start_time = self._convert_to_mountain_time(int(prices[0][0] / 1000))  # Convert ms to seconds
        end_time = self._convert_to_mountain_time(int(prices[-1][0] / 1000))
        
        summary = f"""
{symbol} Historical Summary:
  Period: {start_time} to {end_time}
  Starting Price: ${first_price:,.2f}
  Ending Price: ${last_price:,.2f}
  Change: ${price_change:+,.2f} ({price_change_pct:+.2f}%)
  Highest: ${highest:,.2f}
  Lowest: ${lowest:,.2f}
  Data Points: {len(prices)}"""
        
        return summary.strip()

    def get_historical_bars(self, symbol: str, hours: int = 240) -> 'pd.DataFrame':
        """
        Get historical OHLCV data formatted for technical analysis and LSTM model.
        
        Args:
            symbol: Trading pair in format like BTC/USD
            hours: Number of hours of historical data to retrieve
            
        Returns:
            pandas.DataFrame: Historical OHLCV data with columns [open, high, low, close, volume]
        """
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            print("pandas and numpy are required for get_historical_bars")
            return pd.DataFrame()
        
        # Convert hours to days for CoinGecko API
        days = max(1, hours / 24)
        
        print(f"Fetching {hours} hours ({days:.1f} days) of historical data for {symbol}...")
        
        # Get raw historical data from CoinGecko
        historical_data = self.get_historical_data(symbol, days=int(days))
        
        if not historical_data or 'prices' not in historical_data:
            print(f"No historical data available for {symbol}")
            return pd.DataFrame()
        
        try:
            # Extract data arrays
            prices = historical_data['prices']  # [[timestamp_ms, price], ...]
            volumes = historical_data.get('total_volumes', [])  # [[timestamp_ms, volume], ...]
            market_caps = historical_data.get('market_caps', [])  # [[timestamp_ms, market_cap], ...]
            
            if not prices:
                print(f"No price data available for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
            df_prices.set_index('timestamp', inplace=True)
            
            # Add volumes if available
            if volumes:
                df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
                df_volumes.set_index('timestamp', inplace=True)
                df_prices = df_prices.join(df_volumes, how='left')
            else:
                # If no volume data, create synthetic volume based on price changes
                df_prices['volume'] = np.abs(df_prices['price'].pct_change()) * 1000000
            
            # CoinGecko only provides price points, so we need to create OHLC from price data
            # Resample to hourly data to create proper OHLCV bars
            df_resampled = df_prices.resample('1H').agg({
                'price': ['first', 'max', 'min', 'last'],
                'volume': 'sum'
            }).fillna(method='ffill')
            
            # Flatten column names
            df_resampled.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Fill any remaining NaN values
            df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
            
            # Filter to requested hours
            if len(df_resampled) > hours:
                df_resampled = df_resampled.tail(hours)
            
            print(f"Successfully created {len(df_resampled)} bars for {symbol}")
            print(f"Date range: {df_resampled.index[0]} to {df_resampled.index[-1]}")
            
            return df_resampled
            
        except Exception as e:
            print(f"Error processing historical bars for {symbol}: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    client = CryptoMarketDataClient()
    
    # Get real-time price
    btc_data = client.get_realtime_price("BTC/USD")
    if btc_data:
        print(f"BTC/USD Price: ${btc_data['price']:,.2f}")
        print(f"24h Change: {btc_data['24h_change']:+.2f}%")
        print(f"Last Updated (MT): {btc_data['last_updated_mt']}")
        print(f"Raw Timestamp: {btc_data['last_updated']}")
    
    # Get historical data
    eth_history = client.get_historical_data("ETH/USD", days=7)
    if eth_history:
        summary = client.format_historical_summary(eth_history, "ETH/USD")
        print(summary)
    else:
        print("No historical data available for ETH/USD")
    
    # WebSocket simulation (would need to be run in an async context)
    async def print_data(data):
        print(f"New data for {data['symbol']}: ${data['price']:,.2f} at {data['last_updated_mt']}")
    asyncio.run(client.get_realtime_websocket(['BTC/USD', 'ETH/USD'], print_data))