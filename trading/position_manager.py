import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from data.crypto_data_client import CryptoMarketDataClient
from exchanges.alpaca_client import AlpacaCryptoTrading


class PositionStatus(Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    PENDING = "pending"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionMetrics:
    """Enhanced position metrics for trading decisions."""
    symbol: str
    current_price: float
    entry_price: float
    quantity: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    cost_basis: float
    side: str
    risk_level: RiskLevel
    days_held: int
    volatility: float
    momentum_score: float
    is_profitable: bool
    needs_attention: bool


class PositionManager:
    """
    Manages trading positions with enhanced analytics and risk management.
    Connects CryptoMarketDataClient with AlpacaCryptoTrading for decision making.
    """
    
    def __init__(self, alpaca_client: AlpacaCryptoTrading, data_client: CryptoMarketDataClient):
        """
        Initialize position manager.
        
        Args:
            alpaca_client: Alpaca trading client for execution
            data_client: Crypto data client for position tracking
        """
        self.alpaca_client = alpaca_client
        self.data_client = data_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Position manager initialized")
        
        # Risk management thresholds
        self.risk_thresholds = {
            'stop_loss_pct': 0.05,  # 5% stop loss
            'take_profit_pct': 0.15,  # 15% take profit
            'max_position_age_days': 30,  # Maximum days to hold position
            'volatility_threshold': 0.10,  # 10% volatility threshold
            'correlation_threshold': 0.7  # Portfolio correlation limit
        }
    
    def sync_positions(self) -> Dict:
        """
        Synchronize positions between Alpaca and local data client.
        
        Returns:
            Dictionary containing sync results and discrepancies
        """
        self.logger.info("Synchronizing positions between Alpaca and local data")
        
        try:
            # Get positions from both sources
            alpaca_positions = self.alpaca_client.list_positions()
            local_positions = self.data_client.get_all_positions()
            
            sync_results = {
                'alpaca_count': len(alpaca_positions),
                'local_count': len(local_positions),
                'synced': 0,
                'discrepancies': [],
                'new_positions': [],
                'closed_positions': []
            }
            
            # Convert Alpaca symbols to local format (BTCUSD -> BTC/USD)
            alpaca_symbols = set()
            for pos in alpaca_positions:
                symbol = pos.get('symbol', '')
                if symbol.endswith('USD'):
                    local_symbol = f"{symbol[:-3]}/USD"
                    alpaca_symbols.add(local_symbol)
                    
                    # Update local position if it exists
                    if local_symbol in local_positions:
                        self._update_local_position(local_symbol, pos)
                        sync_results['synced'] += 1
                    else:
                        # Create new local position
                        self._create_local_position(local_symbol, pos)
                        sync_results['new_positions'].append(local_symbol)
            
            # Check for positions that exist locally but not on Alpaca
            for local_symbol in local_positions:
                if local_symbol not in alpaca_symbols and local_symbol != 'closed_positions':
                    sync_results['discrepancies'].append({
                        'symbol': local_symbol,
                        'issue': 'exists_locally_not_alpaca'
                    })
            
            self.logger.info(f"Position sync completed: {sync_results['synced']} synced, "
                           f"{len(sync_results['discrepancies'])} discrepancies")
            
            return sync_results
            
        except Exception as e:
            self.logger.error(f"Failed to sync positions: {str(e)}")
            return {'error': f"Position sync failed: {str(e)}"}
    
    def get_enhanced_position_metrics(self, symbol: str) -> Optional[PositionMetrics]:
        """
        Get comprehensive position metrics for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USD)
            
        Returns:
            PositionMetrics object with enhanced analytics
        """
        try:
            # Get position from data client
            position = self.data_client.get_position(symbol)
            if not position:
                return None
            
            # Get current market data (you may want to implement this in data_client)
            alpaca_symbol = symbol.replace('/', '')  # BTC/USD -> BTCUSD
            try:
                market_data = self.alpaca_client.get_last_trade(alpaca_symbol)
                current_price = float(market_data.get('price', 0))
            except:
                # Fallback to position's last known price
                current_price = position.get('current_price', position.get('entry_price', 0))
            
            # Calculate metrics
            entry_price = position.get('entry_price', 0)
            quantity = position.get('quantity', 0)
            side = position.get('side', 'long')
            
            market_value = current_price * abs(quantity)
            cost_basis = entry_price * abs(quantity)
            
            if side == 'long':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity
            
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            # Calculate additional metrics
            entry_time = datetime.fromisoformat(position.get('entry_time', datetime.now().isoformat()))
            days_held = (datetime.now() - entry_time).days
            
            # Simple volatility calculation (you may want to enhance this)
            volatility = abs(unrealized_pnl_pct) / max(days_held, 1)
            
            # Momentum score (based on recent price movement)
            momentum_score = min(abs(unrealized_pnl_pct) / 10, 1.0)  # Normalized 0-1
            
            # Risk assessment
            risk_level = self._assess_risk_level(unrealized_pnl_pct, days_held, volatility)
            
            return PositionMetrics(
                symbol=symbol,
                current_price=current_price,
                entry_price=entry_price,
                quantity=quantity,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                cost_basis=cost_basis,
                side=side,
                risk_level=risk_level,
                days_held=days_held,
                volatility=volatility,
                momentum_score=momentum_score,
                is_profitable=unrealized_pnl > 0,
                needs_attention=abs(unrealized_pnl_pct) > 25 or days_held > self.risk_thresholds['max_position_age_days']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get enhanced metrics for {symbol}: {str(e)}")
            return None
    
    def get_portfolio_performance(self) -> Dict:
        """
        Get comprehensive portfolio performance metrics.
        
        Returns:
            Dictionary containing portfolio performance data
        """
        try:
            positions = self.data_client.get_all_positions()
            portfolio_metrics = {
                'total_positions': 0,
                'total_market_value': 0,
                'total_unrealized_pnl': 0,
                'total_cost_basis': 0,
                'average_return_pct': 0,
                'risk_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                'positions_by_performance': {'profitable': 0, 'losing': 0},
                'positions_needing_attention': [],
                'top_performers': [],
                'worst_performers': [],
                'portfolio_concentration': {},
                'average_hold_time': 0
            }
            
            position_metrics = []
            total_hold_days = 0
            
            for symbol in positions:
                if symbol == 'closed_positions':
                    continue
                    
                metrics = self.get_enhanced_position_metrics(symbol)
                if metrics:
                    position_metrics.append(metrics)
                    
                    # Update totals
                    portfolio_metrics['total_positions'] += 1
                    portfolio_metrics['total_market_value'] += metrics.market_value
                    portfolio_metrics['total_unrealized_pnl'] += metrics.unrealized_pnl
                    portfolio_metrics['total_cost_basis'] += metrics.cost_basis
                    total_hold_days += metrics.days_held
                    
                    # Risk distribution
                    portfolio_metrics['risk_distribution'][metrics.risk_level.value] += 1
                    
                    # Performance categorization
                    if metrics.is_profitable:
                        portfolio_metrics['positions_by_performance']['profitable'] += 1
                    else:
                        portfolio_metrics['positions_by_performance']['losing'] += 1
                    
                    # Attention needed
                    if metrics.needs_attention:
                        portfolio_metrics['positions_needing_attention'].append({
                            'symbol': metrics.symbol,
                            'reason': 'high_risk_or_old' if metrics.days_held > 20 else 'high_volatility',
                            'pnl_pct': metrics.unrealized_pnl_pct,
                            'days_held': metrics.days_held
                        })
                    
                    # Portfolio concentration
                    concentration_pct = (metrics.market_value / max(portfolio_metrics['total_market_value'], 1)) * 100
                    portfolio_metrics['portfolio_concentration'][symbol] = concentration_pct
            
            # Calculate averages and rankings
            if portfolio_metrics['total_positions'] > 0:
                portfolio_metrics['average_return_pct'] = (
                    portfolio_metrics['total_unrealized_pnl'] / 
                    max(portfolio_metrics['total_cost_basis'], 1) * 100
                )
                portfolio_metrics['average_hold_time'] = total_hold_days / portfolio_metrics['total_positions']
                
                # Sort positions by performance
                position_metrics.sort(key=lambda x: x.unrealized_pnl_pct, reverse=True)
                
                # Top 3 performers
                portfolio_metrics['top_performers'] = [
                    {
                        'symbol': p.symbol,
                        'return_pct': p.unrealized_pnl_pct,
                        'unrealized_pnl': p.unrealized_pnl
                    } for p in position_metrics[:3]
                ]
                
                # Worst 3 performers
                portfolio_metrics['worst_performers'] = [
                    {
                        'symbol': p.symbol,
                        'return_pct': p.unrealized_pnl_pct,
                        'unrealized_pnl': p.unrealized_pnl
                    } for p in position_metrics[-3:]
                ]
            
            self.logger.info(f"Portfolio performance calculated for {portfolio_metrics['total_positions']} positions")
            return portfolio_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio performance: {str(e)}")
            return {'error': f"Portfolio performance calculation failed: {str(e)}"}
    
    def _update_local_position(self, symbol: str, alpaca_position: Dict):
        """Update local position with Alpaca data."""
        try:
            current_price = float(alpaca_position.get('current_price', 0))
            
            position_data = {
                'quantity': float(alpaca_position.get('qty', 0)),
                'current_price': current_price,
                'market_value': float(alpaca_position.get('market_value', 0)),
                'unrealized_pl': float(alpaca_position.get('unrealized_pl', 0)),
                'side': alpaca_position.get('side', 'long'),
                'avg_entry_price': float(alpaca_position.get('avg_entry_price', current_price))
            }
            
            self.data_client.update_position(symbol, position_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update local position {symbol}: {str(e)}")
    
    def _create_local_position(self, symbol: str, alpaca_position: Dict):
        """Create new local position from Alpaca data."""
        try:
            current_price = float(alpaca_position.get('current_price', 0))
            
            position_data = {
                'entry_price': float(alpaca_position.get('avg_entry_price', current_price)),
                'quantity': float(alpaca_position.get('qty', 0)),
                'current_price': current_price,
                'market_value': float(alpaca_position.get('market_value', 0)),
                'unrealized_pl': float(alpaca_position.get('unrealized_pl', 0)),
                'side': alpaca_position.get('side', 'long'),
                'entry_time': datetime.now().isoformat(),
                'source': 'alpaca_sync'
            }
            
            self.data_client.update_position(symbol, position_data)
            
        except Exception as e:
            self.logger.error(f"Failed to create local position {symbol}: {str(e)}")
    
    def _assess_risk_level(self, pnl_pct: float, days_held: int, volatility: float) -> RiskLevel:
        """Assess risk level based on multiple factors."""
        if abs(pnl_pct) > 30 or days_held > 45 or volatility > 5:
            return RiskLevel.CRITICAL
        elif abs(pnl_pct) > 20 or days_held > 30 or volatility > 3:
            return RiskLevel.HIGH
        elif abs(pnl_pct) > 10 or days_held > 14 or volatility > 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW