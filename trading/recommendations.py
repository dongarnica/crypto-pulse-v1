import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from trading.position_manager import PositionManager, PositionMetrics, RiskLevel


class ActionType(Enum):
    HOLD = "hold"
    CLOSE = "close"
    REDUCE = "reduce"
    ADD = "add"
    REBALANCE = "rebalance"
    SET_STOP_LOSS = "set_stop_loss"
    SET_TAKE_PROFIT = "set_take_profit"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TradingRecommendation:
    """Individual trading recommendation."""
    symbol: str
    action: ActionType
    priority: Priority
    confidence: float  # 0-1 scale
    reason: str
    current_price: float
    target_price: Optional[float] = None
    suggested_quantity: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    expected_return: Optional[float] = None
    max_loss: Optional[float] = None
    timeframe: str = "immediate"  # immediate, short_term, medium_term


class TradingRecommendationEngine:
    """
    Generates trading recommendations based on position analysis and market conditions.
    """
    
    def __init__(self, position_manager: PositionManager):
        """
        Initialize recommendation engine.
        
        Args:
            position_manager: Position manager for data access
        """
        self.position_manager = position_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Trading recommendation engine initialized")
        
        # Recommendation parameters
        self.params = {
            'profit_target_pct': 15.0,  # Target profit percentage
            'stop_loss_pct': 5.0,       # Stop loss percentage
            'max_position_age': 30,     # Maximum days to hold
            'rebalance_threshold': 0.1, # Portfolio rebalancing threshold
            'concentration_limit': 0.3, # Maximum position concentration
            'volatility_limit': 0.15    # Maximum acceptable volatility
        }
    
    def generate_recommendations(self, include_portfolio_level: bool = True) -> Dict:
        """
        Generate comprehensive trading recommendations.
        
        Args:
            include_portfolio_level: Whether to include portfolio-level recommendations
            
        Returns:
            Dictionary containing all recommendations organized by priority
        """
        self.logger.info("Generating trading recommendations")
        
        try:
            recommendations = {
                'urgent': [],
                'high': [],
                'medium': [],
                'low': [],
                'portfolio_summary': {},
                'generated_at': datetime.now().isoformat()
            }
            
            # Get portfolio performance
            portfolio_performance = self.position_manager.get_portfolio_performance()
            recommendations['portfolio_summary'] = portfolio_performance
            
            # Generate position-level recommendations
            positions = self.position_manager.data_client.get_all_positions()
            
            for symbol in positions:
                if symbol == 'closed_positions':
                    continue
                
                metrics = self.position_manager.get_enhanced_position_metrics(symbol)
                if metrics:
                    position_recommendations = self._analyze_position(metrics)
                    
                    for rec in position_recommendations:
                        recommendations[rec.priority.name.lower()].append(rec)
            
            # Generate portfolio-level recommendations
            if include_portfolio_level:
                portfolio_recommendations = self._analyze_portfolio(portfolio_performance)
                for rec in portfolio_recommendations:
                    recommendations[rec.priority.name.lower()].append(rec)
            
            # Sort recommendations by confidence within each priority
            for priority in ['urgent', 'high', 'medium', 'low']:
                recommendations[priority].sort(key=lambda x: x.confidence, reverse=True)
            
            total_recommendations = sum(len(recommendations[p]) for p in ['urgent', 'high', 'medium', 'low'])
            self.logger.info(f"Generated {total_recommendations} recommendations")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")
            return {'error': f"Recommendation generation failed: {str(e)}"}
    
    def _analyze_position(self, metrics: PositionMetrics) -> List[TradingRecommendation]:
        """Analyze individual position and generate recommendations."""
        recommendations = []
        
        try:
            # Check for stop loss needs
            if metrics.unrealized_pnl_pct < -self.params['stop_loss_pct']:
                recommendations.append(TradingRecommendation(
                    symbol=metrics.symbol,
                    action=ActionType.CLOSE,
                    priority=Priority.URGENT,
                    confidence=0.9,
                    reason=f"Position down {metrics.unrealized_pnl_pct:.1f}% - stop loss triggered",
                    current_price=metrics.current_price,
                    suggested_quantity=abs(metrics.quantity),
                    max_loss=metrics.unrealized_pnl,
                    timeframe="immediate"
                ))
            
            # Check for profit taking
            elif metrics.unrealized_pnl_pct > self.params['profit_target_pct']:
                recommendations.append(TradingRecommendation(
                    symbol=metrics.symbol,
                    action=ActionType.REDUCE,
                    priority=Priority.HIGH,
                    confidence=0.8,
                    reason=f"Position up {metrics.unrealized_pnl_pct:.1f}% - consider profit taking",
                    current_price=metrics.current_price,
                    suggested_quantity=abs(metrics.quantity) * 0.5,  # Suggest selling half
                    expected_return=metrics.unrealized_pnl * 0.5,
                    timeframe="short_term"
                ))
            
            # Check for position age
            if metrics.days_held > self.params['max_position_age']:
                priority = Priority.HIGH if metrics.days_held > 45 else Priority.MEDIUM
                recommendations.append(TradingRecommendation(
                    symbol=metrics.symbol,
                    action=ActionType.CLOSE,
                    priority=priority,
                    confidence=0.7,
                    reason=f"Position held for {metrics.days_held} days - consider closing",
                    current_price=metrics.current_price,
                    suggested_quantity=abs(metrics.quantity),
                    timeframe="medium_term"
                ))
            
            # Check for high volatility
            if metrics.volatility > self.params['volatility_limit']:
                recommendations.append(TradingRecommendation(
                    symbol=metrics.symbol,
                    action=ActionType.SET_STOP_LOSS,
                    priority=Priority.MEDIUM,
                    confidence=0.6,
                    reason=f"High volatility detected ({metrics.volatility:.1f}%) - set tighter stop loss",
                    current_price=metrics.current_price,
                    target_price=metrics.current_price * (1 - self.params['stop_loss_pct'] / 100),
                    timeframe="immediate"
                ))
            
            # Check for risk level
            if metrics.risk_level == RiskLevel.CRITICAL:
                recommendations.append(TradingRecommendation(
                    symbol=metrics.symbol,
                    action=ActionType.CLOSE,
                    priority=Priority.URGENT,
                    confidence=0.85,
                    reason=f"Critical risk level - immediate action required",
                    current_price=metrics.current_price,
                    suggested_quantity=abs(metrics.quantity),
                    timeframe="immediate"
                ))
            
            # Positive momentum with good profit
            if (metrics.momentum_score > 0.7 and 
                metrics.unrealized_pnl_pct > 5 and 
                metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]):
                
                recommendations.append(TradingRecommendation(
                    symbol=metrics.symbol,
                    action=ActionType.HOLD,
                    priority=Priority.LOW,
                    confidence=0.7,
                    reason=f"Strong momentum ({metrics.momentum_score:.1f}) with good profit - hold position",
                    current_price=metrics.current_price,
                    timeframe="medium_term"
                ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing position {metrics.symbol}: {str(e)}")
        
        return recommendations
    
    def _analyze_portfolio(self, portfolio_performance: Dict) -> List[TradingRecommendation]:
        """Analyze portfolio-level metrics and generate recommendations."""
        recommendations = []
        
        try:
            # Check portfolio concentration
            concentration = portfolio_performance.get('portfolio_concentration', {})
            for symbol, pct in concentration.items():
                if pct > self.params['concentration_limit'] * 100:
                    recommendations.append(TradingRecommendation(
                        symbol=symbol,
                        action=ActionType.REDUCE,
                        priority=Priority.MEDIUM,
                        confidence=0.6,
                        reason=f"Over-concentrated position ({pct:.1f}% of portfolio) - consider reducing",
                        current_price=0,  # Will be filled by position data
                        timeframe="medium_term"
                    ))
            
            # Check overall portfolio performance
            avg_return = portfolio_performance.get('average_return_pct', 0)
            if avg_return < -10:
                recommendations.append(TradingRecommendation(
                    symbol="PORTFOLIO",
                    action=ActionType.REBALANCE,
                    priority=Priority.HIGH,
                    confidence=0.7,
                    reason=f"Portfolio down {avg_return:.1f}% - consider rebalancing",
                    current_price=0,
                    timeframe="short_term"
                ))
            
            # Check for positions needing attention
            attention_needed = portfolio_performance.get('positions_needing_attention', [])
            if len(attention_needed) > 3:
                recommendations.append(TradingRecommendation(
                    symbol="PORTFOLIO",
                    action=ActionType.REBALANCE,
                    priority=Priority.MEDIUM,
                    confidence=0.6,
                    reason=f"{len(attention_needed)} positions need attention - portfolio review needed",
                    current_price=0,
                    timeframe="medium_term"
                ))
            
            # Check risk distribution
            risk_dist = portfolio_performance.get('risk_distribution', {})
            high_risk_count = risk_dist.get('high', 0) + risk_dist.get('critical', 0)
            total_positions = portfolio_performance.get('total_positions', 1)
            
            if high_risk_count / max(total_positions, 1) > 0.4:  # More than 40% high risk
                recommendations.append(TradingRecommendation(
                    symbol="PORTFOLIO",
                    action=ActionType.REBALANCE,
                    priority=Priority.HIGH,
                    confidence=0.8,
                    reason=f"{high_risk_count} high-risk positions out of {total_positions} - reduce portfolio risk",
                    current_price=0,
                    timeframe="immediate"
                ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio: {str(e)}")
        
        return recommendations
    
    def execute_recommendation(self, recommendation: TradingRecommendation) -> Dict:
        """
        Execute a trading recommendation using the Alpaca client.
        
        Args:
            recommendation: The recommendation to execute
            
        Returns:
            Dictionary containing execution results
        """
        self.logger.info(f"Executing recommendation: {recommendation.action.value} for {recommendation.symbol}")
        
        try:
            if recommendation.symbol == "PORTFOLIO":
                return {'status': 'portfolio_action', 'message': 'Portfolio actions require manual review'}
            
            alpaca_symbol = recommendation.symbol.replace('/', '')  # BTC/USD -> BTCUSD
            
            if recommendation.action == ActionType.CLOSE:
                # Close entire position
                position = self.position_manager.alpaca_client.get_position(alpaca_symbol)
                if position:
                    qty = float(position.get('qty', 0))
                    side = 'sell' if float(position.get('qty', 0)) > 0 else 'buy'
                    
                    result = self.position_manager.alpaca_client.place_order(
                        symbol=alpaca_symbol,
                        qty=abs(qty),
                        side=side,
                        type='market'
                    )
                    
                    # Update local position
                    self.position_manager.data_client.close_position(
                        recommendation.symbol,
                        recommendation.current_price,
                        f"Recommendation: {recommendation.reason}"
                    )
                    
                    return {'status': 'executed', 'order': result, 'action': 'close_position'}
            
            elif recommendation.action == ActionType.REDUCE:
                # Reduce position size
                if recommendation.suggested_quantity:
                    position = self.position_manager.alpaca_client.get_position(alpaca_symbol)
                    if position:
                        side = 'sell' if float(position.get('qty', 0)) > 0 else 'buy'
                        
                        result = self.position_manager.alpaca_client.place_order(
                            symbol=alpaca_symbol,
                            qty=recommendation.suggested_quantity,
                            side=side,
                            type='market'
                        )
                        
                        return {'status': 'executed', 'order': result, 'action': 'reduce_position'}
            
            elif recommendation.action == ActionType.SET_STOP_LOSS:
                # Set stop loss order
                if recommendation.target_price:
                    position = self.position_manager.alpaca_client.get_position(alpaca_symbol)
                    if position:
                        qty = abs(float(position.get('qty', 0)))
                        side = 'sell' if float(position.get('qty', 0)) > 0 else 'buy'
                        
                        result = self.position_manager.alpaca_client.place_order(
                            symbol=alpaca_symbol,
                            qty=qty,
                            side=side,
                            type='stop',
                            stop_price=recommendation.target_price
                        )
                        
                        return {'status': 'executed', 'order': result, 'action': 'set_stop_loss'}
            
            return {'status': 'not_implemented', 'message': f'Action {recommendation.action.value} not implemented'}
            
        except Exception as e:
            self.logger.error(f"Failed to execute recommendation: {str(e)}")
            return {'status': 'error', 'message': str(e)}