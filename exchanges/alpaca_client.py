import requests
import time
from typing import Optional, Dict, Union

class AlpacaCryptoTrading:
    """
    A Python module for trading cryptocurrencies using Alpaca's API.
    Supports market orders with separate stop loss and take profit orders.
    """
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        """
        Initialize the Alpaca Crypto Trading client.
        
        Args:
            api_key: Your Alpaca API key
            api_secret: Your Alpaca API secret
            base_url: The base URL for the API (defaults to paper trading)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        })
        
    def _send_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """
        Send a request to the Alpaca API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            Dictionary containing the API response
        """
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, params=params, json=data)
        response.raise_for_status()
        return response.json()
    
    def get_account(self) -> Dict:
        """
        Get the Alpaca account information.
          Returns:
            Dictionary containing account details
        """
        return self._send_request('GET', '/v2/account')
    
    def get_positions(self) -> Dict:
        """
        Get all open positions.
        
        Returns:
            Dictionary containing open positions
        """
        return self._send_request('GET', '/v2/positions')
    
    def list_positions(self) -> list:
        """
        Get all open positions as a list.
        
        Returns:
            List of position dictionaries
        """
        try:
            positions_response = self._send_request('GET', '/v2/positions')
            # If the response is already a list, return it
            if isinstance(positions_response, list):
                return positions_response
            # If it's a dict with a 'positions' key, return that
            elif isinstance(positions_response, dict) and 'positions' in positions_response:
                return positions_response['positions']
            # If it's some other dict structure, try to convert it
            elif isinstance(positions_response, dict):
                return [positions_response]
            else:
                return []
        except Exception as e:
            print(f"Error fetching positions: {str(e)}")
            return []
    
    def get_position(self, symbol: str) -> Dict:
        """
        Get a specific position.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., BTCUSD)
            
        Returns:
            Dictionary containing position details
        """
        return self._send_request('GET', f'/v2/positions/{symbol}')
    
    def get_assets(self) -> Dict:
        """
        Get all tradable assets.
        
        Returns:
            Dictionary containing tradable assets
        """
        return self._send_request('GET', '/v2/assets')
    
    def get_asset(self, symbol: str) -> Dict:
        """
        Get a specific asset.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., BTCUSD)
            
        Returns:
            Dictionary containing asset details
        """
        return self._send_request('GET', f'/v2/assets/{symbol}')
    
    def get_last_trade(self, symbol: str) -> Dict:
        """
        Get the last trade for a symbol.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., BTCUSD)
            
        Returns:
            Dictionary containing last trade details
        """
        return self._send_request('GET', f'/v1/last/crypto/{symbol}')
    
    def get_last_quote(self, symbol: str) -> Dict:
        """
        Get the last quote for a symbol.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., BTCUSD)
            
        Returns:
            Dictionary containing last quote details
        """
        return self._send_request('GET', f'/v1/last_quote/crypto/{symbol}')
    
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        type: str = 'market',
        time_in_force: str = 'gtc',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict:
        """
        Place a new order.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., BTCUSD)
            qty: The quantity to trade
            side: 'buy' or 'sell'
            type: Order type (default: 'market')
            time_in_force: Time in force (default: 'gtc')
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            
        Returns:
            Dictionary containing order details
        """
        data = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': type,
            'time_in_force': time_in_force,
        }
        
        if limit_price is not None:
            data['limit_price'] = str(limit_price)
        if stop_price is not None:
            data['stop_price'] = str(stop_price)
            
        return self._send_request('POST', '/v2/orders', data=data)
    
    def get_order(self, order_id: str) -> Dict:
        """
        Get an order by ID.
        
        Args:
            order_id: The order ID
            
        Returns:
            Dictionary containing order details
        """
        return self._send_request('GET', f'/v2/orders/{order_id}')
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order by ID.
        
        Args:
            order_id: The order ID
            
        Returns:
            Empty dictionary if successful
        """
        return self._send_request('DELETE', f'/v2/orders/{order_id}')
    
    def get_orders(self, status: Optional[str] = None) -> Dict:
        """
        Get all orders.
        
        Args:
            status: Filter by status (open, closed, all)
            
        Returns:
            Dictionary containing orders
        """
        params = {}
        if status is not None:
            params['status'] = status
        return self._send_request('GET', '/v2/orders', params=params)
    
    def trade_with_limits(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        wait_for_fill: bool = True,
        wait_timeout: int = 30
    ) -> Dict:
        """
        Execute a trade with separate stop loss and take profit orders.
        This avoids using bracket orders by creating the orders separately.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., BTCUSD)
            qty: The quantity to trade
            side: 'buy' or 'sell'
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            wait_for_fill: Whether to wait for the main order to fill before placing SL/TP
            wait_timeout: Timeout in seconds to wait for order to fill
            
        Returns:
            Dictionary containing the main order and SL/TP order details
        """
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be either 'buy' or 'sell'")
            
        if stop_loss is None and take_profit is None:
            raise ValueError("At least one of stop_loss or take_profit must be specified")
            
        # Place the main market order
        main_order = self.place_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market'
        )
        
        result = {
            'main_order': main_order,
            'stop_loss_order': None,
            'take_profit_order': None
        }
        
        # If we need to wait for the main order to fill
        if wait_for_fill:
            start_time = time.time()
            while True:
                order_status = self.get_order(main_order['id'])['status']
                if order_status == 'filled':
                    break
                if time.time() - start_time > wait_timeout:
                    raise TimeoutError("Timed out waiting for order to fill")
                time.sleep(1)
        
        # Determine the opposite side for SL/TP orders
        opposite_side = 'sell' if side == 'buy' else 'buy'
        
        # Place stop loss order if specified
        if stop_loss is not None:
            stop_order = self.place_order(
                symbol=symbol,
                qty=qty,
                side=opposite_side,
                type='stop',
                stop_price=stop_loss,
                time_in_force='gtc'
            )
            result['stop_loss_order'] = stop_order
        
        # Place take profit order if specified
        if take_profit is not None:
            take_profit_order = self.place_order(
                symbol=symbol,
                qty=qty,
                side=opposite_side,
                type='limit',
                limit_price=take_profit,
                time_in_force='gtc'
            )
            result['take_profit_order'] = take_profit_order
            
        return result
    
    def cancel_all_orders(self) -> Dict:
        """
        Cancel all open orders.
        
        Returns:
            Empty dictionary if successful
        """
        return self._send_request('DELETE', '/v2/orders')
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get a comprehensive portfolio summary including positions analysis.
        
        Returns:
            Dictionary containing portfolio summary with analysis
        """
        try:
            # Get account info
            account = self.get_account()
            positions = self.list_positions()
            
            # Initialize summary
            summary = {
                'account': {
                    'cash': float(account.get('cash', 0)),
                    'buying_power': float(account.get('buying_power', 0)),
                    'portfolio_value': float(account.get('portfolio_value', 0)),
                    'status': account.get('status', 'Unknown')
                },
                'positions': {
                    'count': len(positions) if positions else 0,
                    'total_market_value': 0,
                    'total_unrealized_pl': 0,
                    'total_cost_basis': 0,
                    'by_symbol': {}
                },
                'performance': {
                    'total_return_pct': 0,
                    'best_performer': None,
                    'worst_performer': None
                }
            }
            
            if positions:
                best_pct = float('-inf')
                worst_pct = float('inf')
                
                for pos in positions:
                    symbol = pos.get('symbol', 'N/A')
                    try:
                        market_value = float(pos.get('market_value', 0))
                        unrealized_pl = float(pos.get('unrealized_pl', 0))
                        cost_basis = float(pos.get('cost_basis', 0))
                        unrealized_plpc = float(pos.get('unrealized_plpc', 0))
                        
                        # Add to totals
                        summary['positions']['total_market_value'] += market_value
                        summary['positions']['total_unrealized_pl'] += unrealized_pl
                        summary['positions']['total_cost_basis'] += cost_basis
                        
                        # Track best/worst performers
                        if unrealized_plpc > best_pct:
                            best_pct = unrealized_plpc
                            summary['performance']['best_performer'] = {
                                'symbol': symbol,
                                'return_pct': unrealized_plpc * 100,
                                'unrealized_pl': unrealized_pl
                            }
                        
                        if unrealized_plpc < worst_pct:
                            worst_pct = unrealized_plpc
                            summary['performance']['worst_performer'] = {
                                'symbol': symbol,
                                'return_pct': unrealized_plpc * 100,
                                'unrealized_pl': unrealized_pl
                            }
                        
                        # Store individual position data
                        summary['positions']['by_symbol'][symbol] = {
                            'qty': float(pos.get('qty', 0)),
                            'market_value': market_value,
                            'unrealized_pl': unrealized_pl,
                            'return_pct': unrealized_plpc * 100,
                            'cost_basis': cost_basis
                        }
                        
                    except (ValueError, TypeError):
                        continue
                
                # Calculate total return percentage
                if summary['positions']['total_cost_basis'] > 0:
                    summary['performance']['total_return_pct'] = (
                        summary['positions']['total_unrealized_pl'] / 
                        summary['positions']['total_cost_basis']
                    ) * 100
            
            return summary
            
        except Exception as e:
            return {'error': f"Failed to get portfolio summary: {str(e)}"}
    
    def analyze_position_risk(self, symbol: str) -> Dict:
        """
        Analyze risk metrics for a specific position.
        
        Args:
            symbol: The cryptocurrency symbol to analyze
            
        Returns:
            Dictionary containing risk analysis
        """
        try:
            position = self.get_position(symbol)
            if not position:
                return {'error': f'No position found for {symbol}'}
            
            current_price = float(position.get('current_price', 0))
            avg_entry_price = float(position.get('avg_entry_price', 0))
            market_value = float(position.get('market_value', 0))
            unrealized_pl = float(position.get('unrealized_pl', 0))
            
            # Calculate basic risk metrics
            price_change_pct = ((current_price - avg_entry_price) / avg_entry_price * 100) if avg_entry_price > 0 else 0
            
            # Risk levels based on unrealized P&L percentage
            risk_level = 'Low'
            if abs(price_change_pct) > 20:
                risk_level = 'High'
            elif abs(price_change_pct) > 10:
                risk_level = 'Medium'
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'entry_price': avg_entry_price,
                'market_value': market_value,
                'unrealized_pl': unrealized_pl,
                'price_change_pct': price_change_pct,
                'risk_level': risk_level,
                'analysis': {
                    'is_profitable': unrealized_pl > 0,
                    'is_high_risk': abs(price_change_pct) > 15,
                    'needs_attention': abs(price_change_pct) > 25
                }
            }
            
        except Exception as e:
            return {'error': f"Failed to analyze position risk for {symbol}: {str(e)}"}