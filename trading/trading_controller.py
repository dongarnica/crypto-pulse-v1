import logging
from typing import Dict, List, Optional
from datetime import datetime

from config.config import AppConfig
from exchanges.alpaca_client import AlpacaCryptoTrading
from data.crypto_data_client import CryptoMarketDataClient
from trading.position_manager import PositionManager
from trading.recommendations import TradingRecommendationEngine, TradingRecommendation, Priority
from trading.performance_analytics import PerformanceAnalytics


class TradingController:
    """
    Main controller that orchestrates position management, recommendations, and performance analytics.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize trading controller.
        
        Args:
            config: Application configuration (creates new if None)
        """
        self.config = config or AppConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize clients
        alpaca_config = self.config.get_alpaca_config()
        self.alpaca_client = AlpacaCryptoTrading(
            api_key=alpaca_config['api_key'],
            api_secret=alpaca_config['api_secret'],
            base_url=alpaca_config['base_url']
        )
        
        self.data_client = CryptoMarketDataClient()
        
        # Initialize trading components
        self.position_manager = PositionManager(self.alpaca_client, self.data_client)
        self.recommendation_engine = TradingRecommendationEngine(self.position_manager)
        self.performance_analytics = PerformanceAnalytics(self.position_manager)
        
        self.logger.info("Trading controller initialized")
    
    def run_daily_analysis(self) -> Dict:
        """
        Run comprehensive daily analysis including sync, recommendations, and performance tracking.
        
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting daily trading analysis")
        
        try:
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'sync_results': {},
                'recommendations': {},
                'performance_snapshot': {},
                'performance_report': {},
                'actions_taken': [],
                'summary': {}
            }
            
            # 1. Sync positions
            self.logger.info("Syncing positions...")
            sync_results = self.position_manager.sync_positions()
            analysis_results['sync_results'] = sync_results
            
            # 2. Generate recommendations
            self.logger.info("Generating recommendations...")
            recommendations = self.recommendation_engine.generate_recommendations()
            analysis_results['recommendations'] = recommendations
            
            # 3. Track daily performance
            self.logger.info("Tracking performance...")
            performance_snapshot = self.performance_analytics.track_daily_performance()
            analysis_results['performance_snapshot'] = performance_snapshot
            
            # 4. Generate performance report
            performance_report = self.performance_analytics.generate_performance_report()
            analysis_results['performance_report'] = performance_report
            
            # 5. Execute urgent recommendations (optional)
            urgent_recommendations = recommendations.get('urgent', [])
            if urgent_recommendations:
                self.logger.info(f"Found {len(urgent_recommendations)} urgent recommendations")
                # Note: Auto-execution is commented out for safety
                # analysis_results['actions_taken'] = self._execute_urgent_actions(urgent_recommendations)
            
            # 6. Generate summary
            analysis_results['summary'] = self._generate_analysis_summary(analysis_results)
            
            self.logger.info("Daily analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Daily analysis failed: {str(e)}")
            return {'error': f"Daily analysis failed: {str(e)}"}
    
    def get_portfolio_dashboard(self) -> Dict:
        """
        Get comprehensive portfolio dashboard data.
        
        Returns:
            Dictionary containing dashboard information
        """
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_overview': {},
                'position_metrics': [],
                'recommendations_summary': {},
                'performance_metrics': {},
                'risk_analysis': {}
            }
            
            # Portfolio overview
            dashboard['portfolio_overview'] = self.position_manager.get_portfolio_performance()
            
            # Individual position metrics
            positions = self.data_client.get_all_positions()
            for symbol in positions:
                if symbol != 'closed_positions':
                    metrics = self.position_manager.get_enhanced_position_metrics(symbol)
                    if metrics:
                        dashboard['position_metrics'].append({
                            'symbol': metrics.symbol,
                            'market_value': metrics.market_value,
                            'unrealized_pnl': metrics.unrealized_pnl,
                            'unrealized_pnl_pct': metrics.unrealized_pnl_pct,
                            'risk_level': metrics.risk_level.value,
                            'days_held': metrics.days_held,
                            'is_profitable': metrics.is_profitable,
                            'needs_attention': metrics.needs_attention
                        })
            
            # Recommendations summary
            recommendations = self.recommendation_engine.generate_recommendations()
            dashboard['recommendations_summary'] = {
                'urgent_count': len(recommendations.get('urgent', [])),
                'high_count': len(recommendations.get('high', [])),
                'medium_count': len(recommendations.get('medium', [])),
                'low_count': len(recommendations.get('low', []))
            }
            
            # Performance metrics
            performance_metrics = self.performance_analytics.calculate_portfolio_metrics()
            dashboard['performance_metrics'] = {
                'total_return_pct': performance_metrics.total_return_pct,
                'win_rate': performance_metrics.win_rate,
                'sharpe_ratio': performance_metrics.sharpe_ratio,
                'max_drawdown_pct': performance_metrics.max_drawdown_pct,
                'total_trades': performance_metrics.total_trades
            }
            
            # Risk analysis
            risk_dist = dashboard['portfolio_overview'].get('risk_distribution', {})
            total_positions = dashboard['portfolio_overview'].get('total_positions', 1)
            dashboard['risk_analysis'] = {
                'low_risk_pct': (risk_dist.get('low', 0) / max(total_positions, 1)) * 100,
                'medium_risk_pct': (risk_dist.get('medium', 0) / max(total_positions, 1)) * 100,
                'high_risk_pct': (risk_dist.get('high', 0) / max(total_positions, 1)) * 100,
                'critical_risk_pct': (risk_dist.get('critical', 0) / max(total_positions, 1)) * 100,
                'positions_needing_attention': len(dashboard['portfolio_overview'].get('positions_needing_attention', []))
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Failed to get portfolio dashboard: {str(e)}")
            return {'error': f"Dashboard generation failed: {str(e)}"}
    
    def execute_recommendation_by_id(self, recommendation_data: Dict) -> Dict:
        """
        Execute a specific recommendation.
        
        Args:
            recommendation_data: Dictionary containing recommendation details
            
        Returns:
            Dictionary containing execution results
        """
        try:
            # Convert dict back to recommendation object
            from trading.recommendations import ActionType, Priority
            
            recommendation = TradingRecommendation(
                symbol=recommendation_data['symbol'],
                action=ActionType(recommendation_data['action']),
                priority=Priority(recommendation_data['priority']),
                confidence=recommendation_data['confidence'],
                reason=recommendation_data['reason'],
                current_price=recommendation_data['current_price'],
                target_price=recommendation_data.get('target_price'),
                suggested_quantity=recommendation_data.get('suggested_quantity'),
                timeframe=recommendation_data.get('timeframe', 'immediate')
            )
            
            result = self.recommendation_engine.execute_recommendation(recommendation)
            
            self.logger.info(f"Executed recommendation for {recommendation.symbol}: {result.get('status')}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute recommendation: {str(e)}")
            return {'error': f"Recommendation execution failed: {str(e)}"}
    
    def _execute_urgent_actions(self, urgent_recommendations: List[Dict]) -> List[Dict]:
        """Execute urgent recommendations automatically (use with caution)."""
        actions_taken = []
        
        for rec_data in urgent_recommendations:
            try:
                result = self.execute_recommendation_by_id(rec_data)
                actions_taken.append({
                    'recommendation': rec_data,
                    'result': result,
                    'executed_at': datetime.now().isoformat()
                })
            except Exception as e:
                actions_taken.append({
                    'recommendation': rec_data,
                    'result': {'error': str(e)},
                    'executed_at': datetime.now().isoformat()
                })
        
        return actions_taken
    
    def _generate_analysis_summary(self, analysis_results: Dict) -> Dict:
        """Generate summary of analysis results."""
        try:
            sync_results = analysis_results.get('sync_results', {})
            recommendations = analysis_results.get('recommendations', {})
            performance = analysis_results.get('performance_snapshot', {})
            
            summary = {
                'positions_synced': sync_results.get('synced', 0),
                'new_positions': len(sync_results.get('new_positions', [])),
                'discrepancies': len(sync_results.get('discrepancies', [])),
                'total_recommendations': sum(len(recommendations.get(p, [])) for p in ['urgent', 'high', 'medium', 'low']),
                'urgent_actions_needed': len(recommendations.get('urgent', [])),
                'portfolio_value': performance.get('portfolio_value', 0),
                'unrealized_pnl': performance.get('unrealized_pnl', 0),
                'total_positions': performance.get('total_positions', 0),
                'analysis_status': 'completed'
            }
            
            # Generate key messages
            messages = []
            
            if summary['urgent_actions_needed'] > 0:
                messages.append(f"{summary['urgent_actions_needed']} urgent actions needed")
            
            if summary['discrepancies'] > 0:
                messages.append(f"{summary['discrepancies']} position discrepancies found")
            
            if summary['unrealized_pnl'] < 0:
                messages.append(f"Portfolio has unrealized losses: ${summary['unrealized_pnl']:.2f}")
            elif summary['unrealized_pnl'] > 0:
                messages.append(f"Portfolio has unrealized gains: ${summary['unrealized_pnl']:.2f}")
            
            summary['key_messages'] = messages
            
            return summary
            
        except Exception as e:
            return {'error': f"Summary generation failed: {str(e)}"}