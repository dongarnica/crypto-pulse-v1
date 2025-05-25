import os
import json
import time
import logging
import requests
import asyncio
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, Optional, Union, List

class CryptoMarketDataClient:
    """
    Provides historical and real-time crypto market data using CoinGecko API.
    Supports symbol notation like BTC/USD.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        load_dotenv()
        self.mountain_tz = pytz.timezone('America/Denver')  # Mountain Time
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Crypto market data client initialized")
        self.logger.debug(f"Using timezone: {self.mountain_tz}")
        
        # Initialize positions tracking
        self.positions_file = os.path.join(os.path.dirname(__file__), 'positions.json')
        self.positions = self._load_positions()
        
    def _load_positions(self) -> Dict:
        """Load positions from file or initialize empty positions"""
        try:
            if os.path.exists(self.positions_file):
                with open(self.positions_file, 'r') as f:
                    positions = json.load(f)
                    self.logger.info(f"Loaded {len(positions)} positions from file")
                    return positions
        except Exception as e:
            self.logger.warning(f"Error loading positions file: {e}")
        
        # Return empty positions structure
        return {}
    
    def _save_positions(self):
        """Save current positions to file"""
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(self.positions, f, indent=2, default=str)
            self.logger.debug("Positions saved to file")
        except Exception as e:
            self.logger.error(f"Error saving positions: {e}")
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position for a symbol
        
        Args:
            symbol: Trading pair in format like BTC/USD
            
        Returns:
            dict: Position data or None if no position
        """
        return self.positions.get(symbol)
    
    def update_position(self, symbol: str, position_data: Dict):
        """
        Update position for a symbol
        
        Args:
            symbol: Trading pair in format like BTC/USD
            position_data: Position information
        """
        self.positions[symbol] = {
            **position_data,
            'last_updated': datetime.now().isoformat(),
            'symbol': symbol
        }
        self._save_positions()
        self.logger.info(f"Updated position for {symbol}: {position_data}")
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str = "manual"):
        """
        Close a position
        
        Args:
            symbol: Trading pair
            exit_price: Price at which position was closed
            exit_reason: Reason for closing
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Calculate P&L
            entry_price = position.get('entry_price', exit_price)
            quantity = position.get('quantity', 0)
            side = position.get('side', 'long')
            
            if side == 'long':
                pnl = (exit_price - entry_price) * quantity
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:  # short
                pnl = (entry_price - exit_price) * quantity
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            
            # Archive the position
            closed_position = {
                **position,
                'exit_price': exit_price,
                'exit_time': datetime.now().isoformat(),
                'exit_reason': exit_reason,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'status': 'closed'
            }
            
            # Move to closed positions
            if 'closed_positions' not in self.positions:
                self.positions['closed_positions'] = []
            self.positions['closed_positions'].append(closed_position)
            
            # Remove from active positions
            del self.positions[symbol]
            self._save_positions()
            
            self.logger.info(f"Closed position for {symbol}: P&L = {pnl:.2f} ({pnl_pct:+.2f}%)")
            return closed_position
        
        return None
    
    def get_all_positions(self) -> Dict:
        """Get all active positions"""
        active_positions = {k: v for k, v in self.positions.items() 
                          if k != 'closed_positions' and isinstance(v, dict)}
        return active_positions
    
    def get_position_summary(self, symbol: str = None) -> Dict:
        """
        Get position summary for a symbol or all positions
        
        Args:
            symbol: Optional symbol to get summary for specific position
            
        Returns:
            dict: Position summary with P&L, exposure, etc.
        """
        if symbol:
            position = self.get_position(symbol)
            if not position:
                return {'status': 'no_position', 'symbol': symbol}
            
            # Get current price for P&L calculation
            current_data = self.get_realtime_price(symbol)
            current_price = current_data['price'] if current_data else position.get('entry_price', 0)
            
            entry_price = position.get('entry_price', current_price)
            quantity = position.get('quantity', 0)
            side = position.get('side', 'long')
            
            # Calculate unrealized P&L
            if side == 'long':
                unrealized_pnl = (current_price - entry_price) * quantity
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # short
                unrealized_pnl = (entry_price - current_price) * quantity
                unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            return {
                'symbol': symbol,
                'status': 'active',
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'entry_time': position.get('entry_time'),
                'stop_loss': position.get('stop_loss'),
                'take_profit': position.get('take_profit'),
                'risk_amount': position.get('risk_amount', 0),
                'position_value': current_price * quantity
            }
        else:
            # Summary for all positions
            all_positions = self.get_all_positions()
            summary = {
                'total_positions': len(all_positions),
                'long_positions': 0,
                'short_positions': 0,
                'total_exposure': 0,
                'total_unrealized_pnl': 0,
                'positions': {}
            }
            
            for sym, pos in all_positions.items():
                pos_summary = self.get_position_summary(sym)
                if pos_summary['status'] == 'active':
                    summary['positions'][sym] = pos_summary
                    summary['total_exposure'] += pos_summary['position_value']
                    summary['total_unrealized_pnl'] += pos_summary['unrealized_pnl']
                    
                    if pos_summary['side'] == 'long':
                        summary['long_positions'] += 1
                    else:
                        summary['short_positions'] += 1
            
            return summary

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
        self.logger.info(f"Fetching real-time price for {symbol} (coin_id: {coin_id}, vs_currency: {vs_currency})")
        
        endpoint = "/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': vs_currency,
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        start_time = time.time()
        try:
            self.logger.debug(f"Making API request to {self.BASE_URL}{endpoint} with params: {params}")
            
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            elapsed_time = time.time() - start_time
            
            if coin_id not in data:
                self.logger.warning(f"No data found for {coin_id}")
                return None
            
            coin_data = data[coin_id]
            price = coin_data.get(vs_currency)
            market_cap = coin_data.get(f"{vs_currency}_market_cap")
            vol_24h = coin_data.get(f"{vs_currency}_24h_vol")
            change_24h = coin_data.get(f"{vs_currency}_24h_change")
            last_updated = coin_data.get("last_updated_at")
            
            result = {
                'symbol': symbol,
                'price': price,
                'market_cap': market_cap,
                'volume_24h': vol_24h,
                'change_24h': change_24h,
                'last_updated': self._convert_to_mountain_time(last_updated),
                'timestamp': datetime.now().isoformat(),
                'fetch_time_seconds': round(elapsed_time, 3)
            }
            
            self.logger.info(f"Successfully fetched real-time data for {symbol} in {elapsed_time:.3f}s")
            return result
            
        except requests.exceptions.RequestException as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Request error fetching real-time data for {symbol} after {elapsed_time:.2f}s: {e}")
            return None
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Error fetching real-time data for {symbol} after {elapsed_time:.2f}s: {e}")
            return None

    async def get_realtime_websocket(self, symbols: list, callback: callable):
        """
        Mock WebSocket implementation - would need actual WebSocket library
        """
        self.logger.info(f"Mock WebSocket connection for symbols: {symbols}")
        
        while True:
            for symbol in symbols:
                data = self.get_realtime_price(symbol)
                if data:
                    await callback(data)
            await asyncio.sleep(5)  # Update every 5 seconds

    def get_historical_data(self, symbol: str, days: int = 1) -> Optional[Dict]:
        """
        Get historical price data for a cryptocurrency pair.
        
        Args:
            symbol: Trading pair in format like BTC/USD
            days: Number of days of historical data
            
        Returns:
            dict: Historical market data
        """
        coin_id, vs_currency = self._normalize_symbol(symbol)
        self.logger.info(f"Fetching {days} days of historical data for {symbol}")
        
        endpoint = "/coins/{id}/market_chart".format(id=coin_id)
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        
        start_time = time.time()
        try:
            self.logger.debug(f"Making historical data request to {self.BASE_URL}{endpoint} with params: {params}")
            
            response = requests.get(f"{self.BASE_URL}{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            elapsed_time = time.time() - start_time
            
            # Parse the data
            prices = data.get('prices', [])
            market_caps = data.get('market_caps', [])
            total_volumes = data.get('total_volumes', [])
            
            if not prices:
                self.logger.warning(f"No price data found for {symbol}")
                return None
            
            result = {
                'symbol': symbol,
                'prices': prices,
                'market_caps': market_caps,
                'total_volumes': total_volumes,
                'days': days,
                'data_points': len(prices),
                'start_time': self._convert_to_mountain_time(prices[0][0] // 1000) if prices else None,
                'end_time': self._convert_to_mountain_time(prices[-1][0] // 1000) if prices else None,
                'fetch_time_seconds': round(elapsed_time, 3)
            }
            
            self.logger.info(f"Successfully fetched {len(prices)} historical data points for {symbol} in {elapsed_time:.3f}s")
            return result
            
        except requests.exceptions.RequestException as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Request error fetching historical data for {symbol} after {elapsed_time:.2f}s: {e}")
            return None
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Error fetching historical data for {symbol} after {elapsed_time:.2f}s: {e}")
            return None

    def format_historical_summary(self, historical_data: Dict, symbol: str) -> str:
        """Format historical data into a readable summary"""
        if not historical_data:
            return f"No historical data available for {symbol}"
        
        prices = historical_data.get('prices', [])
        if not prices:
            return f"No price data available for {symbol}"
        
        first_price = prices[0][1]
        last_price = prices[-1][1]
        change = ((last_price - first_price) / first_price) * 100
        
        high_price = max(price[1] for price in prices)
        low_price = min(price[1] for price in prices)
        
        return f"""
📊 Historical Summary for {symbol}:
• Period: {historical_data['start_time']} to {historical_data['end_time']}
• Price Change: ${first_price:.2f} → ${last_price:.2f} ({change:+.2f}%)
• High: ${high_price:.2f}
• Low: ${low_price:.2f}
• Data Points: {len(prices)}
"""

    def get_historical_bars(self, symbol: str, hours: int = 240):
        """
        Get historical OHLCV data formatted for trading analysis
        
        Args:
            symbol: Trading pair symbol
            hours: Number of hours of data to fetch
            
        Returns:
            DataFrame: OHLCV data with technical indicators
        """
        print(f"Fetching {hours} hours ({hours/24:.1f} days) of historical data for {symbol}...")
        
        # Convert hours to days for the API call
        days = max(1, hours / 24)
        
        historical_data = self.get_historical_data(symbol, days=days)
        
        if not historical_data or not historical_data.get('prices'):
            print(f"No historical data available for {symbol}")
            return pd.DataFrame()
        
        try:
            # Extract price and volume data
            prices = historical_data['prices']
            volumes = historical_data.get('total_volumes', [])
            
            # Create DataFrame
            df_data = []
            
            for i, (timestamp, price) in enumerate(prices):
                volume = volumes[i][1] if i < len(volumes) else 0
                
                df_data.append({
                    'timestamp': pd.to_datetime(timestamp, unit='ms'),
                    'open': price,  # Simplified - using same price for all OHLC
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Filter to the requested number of hours
            if hours < 24 * days:
                cutoff_time = df.index[-1] - pd.Timedelta(hours=hours)
                df = df[df.index >= cutoff_time]
            
            # Add some basic technical indicators
            if len(df) > 20:
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            print(f"✅ Successfully processed {len(df)} data points for {symbol}")
            print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing historical bars for {symbol}: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    client = CryptoMarketDataClient()
    
    # Get real-time price
    btc_data = client.get_realtime_price("BTC/USD")
    if btc_data:
        print(f"💰 BTC/USD: ${btc_data['price']:,.2f}")
        print(f"📈 24h Change: {btc_data['change_24h']:+.2f}%")
    
    # Get historical data
    eth_history = client.get_historical_data("ETH/USD", days=7)
    if eth_history:
        summary = client.format_historical_summary(eth_history, "ETH/USD")
        print(summary)
    else:
        print("❌ Failed to fetch ETH historical data")
    
    # Get historical bars for LSTM
    bars = client.get_historical_bars("BTC/USD", hours=168)
    if not bars.empty:
        print(f"📊 Retrieved {len(bars)} hourly bars for BTC/USD")
    else:
        print("❌ Failed to fetch BTC historical bars")