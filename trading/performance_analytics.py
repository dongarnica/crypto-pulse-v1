import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from trading.position_manager import PositionManager


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_trades: int
    profitable_trades: int
    losing_trades: int
    avg_hold_time_days: float
    volatility: float
    calmar_ratio: float


class PerformanceAnalytics:
    """
    Advanced performance analytics for trading positions and portfolio.
    """
    
    def __init__(self, position_manager: PositionManager):
        """
        Initialize performance analytics.
        
        Args:
            position_manager: Position manager for data access
        """
        self.position_manager = position_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Performance analytics initialized")
    
    def calculate_portfolio_metrics(self, period_days: int = 30) -> PerformanceMetrics:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            period_days: Analysis period in days
            
        Returns:
            PerformanceMetrics object with calculated metrics
        """
        self.logger.info(f"Calculating portfolio metrics for {period_days} days")
        
        try:
            # Get closed positions for analysis
            positions_data = self.position_manager.data_client.positions
            closed_positions = positions_data.get('closed_positions', [])
            
            if not closed_positions:
                return self._empty_metrics()
            
            # Filter positions by period
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_trades = []
            
            for trade in closed_positions:
                exit_time = datetime.fromisoformat(trade.get('exit_time', datetime.now().isoformat()))
                if exit_time >= cutoff_date:
                    recent_trades.append(trade)
            
            if not recent_trades:
                return self._empty_metrics()
            
            # Calculate basic metrics
            total_trades = len(recent_trades)
            profitable_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            losing_trades = total_trades - profitable_trades
            
            returns = [trade.get('pnl_pct', 0) for trade in recent_trades]
            win_returns = [r for r in returns if r > 0]
            loss_returns = [r for r in returns if r < 0]
            
            # Performance calculations
            total_return_pct = sum(returns)
            avg_return = sum(returns) / len(returns) if returns else 0
            volatility = self._calculate_volatility(returns)
            
            # Annualized return (assuming 365 trading days)
            days_covered = (datetime.now() - cutoff_date).days
            annualized_return_pct = (avg_return * 365 / max(days_covered, 1)) if days_covered > 0 else 0
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 2.0
            sharpe_ratio = ((annualized_return_pct - risk_free_rate) / volatility) if volatility > 0 else 0
            
            # Win rate and average win/loss
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            avg_win_pct = sum(win_returns) / len(win_returns) if win_returns else 0
            avg_loss_pct = sum(loss_returns) / len(loss_returns) if loss_returns else 0
            
            # Profit factor
            total_wins = sum(trade.get('pnl', 0) for trade in recent_trades if trade.get('pnl', 0) > 0)
            total_losses = abs(sum(trade.get('pnl', 0) for trade in recent_trades if trade.get('pnl', 0) < 0))
            profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
            
            # Average hold time
            hold_times = []
            for trade in recent_trades:
                entry_time = datetime.fromisoformat(trade.get('entry_time', datetime.now().isoformat()))
                exit_time = datetime.fromisoformat(trade.get('exit_time', datetime.now().isoformat()))
                hold_days = (exit_time - entry_time).days
                hold_times.append(hold_days)
            
            avg_hold_time_days = sum(hold_times) / len(hold_times) if hold_times else 0
            
            # Max drawdown
            max_drawdown_pct = self._calculate_max_drawdown(returns)
            
            # Calmar ratio
            calmar_ratio = (annualized_return_pct / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else 0
            
            return PerformanceMetrics(
                total_return_pct=total_return_pct,
                annualized_return_pct=annualized_return_pct,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_pct=max_drawdown_pct,
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                profit_factor=profit_factor,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                losing_trades=losing_trades,
                avg_hold_time_days=avg_hold_time_days,
                volatility=volatility,
                calmar_ratio=calmar_ratio
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio metrics: {str(e)}")
            return self._empty_metrics()
    
    def generate_performance_report(self, period_days: int = 30) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            period_days: Analysis period in days
            
        Returns:
            Dictionary containing detailed performance analysis
        """
        self.logger.info(f"Generating performance report for {period_days} days")
        
        try:
            # Get performance metrics
            metrics = self.calculate_portfolio_metrics(period_days)
            
            # Get current portfolio status
            portfolio_performance = self.position_manager.get_portfolio_performance()
            
            # Generate report
            report = {
                'report_generated': datetime.now().isoformat(),
                'analysis_period_days': period_days,
                'performance_metrics': {
                    'total_return_pct': round(metrics.total_return_pct, 2),
                    'annualized_return_pct': round(metrics.annualized_return_pct, 2),
                    'sharpe_ratio': round(metrics.sharpe_ratio, 2),
                    'max_drawdown_pct': round(metrics.max_drawdown_pct, 2),
                    'win_rate': round(metrics.win_rate, 2),
                    'avg_win_pct': round(metrics.avg_win_pct, 2),
                    'avg_loss_pct': round(metrics.avg_loss_pct, 2),
                    'profit_factor': round(metrics.profit_factor, 2),
                    'volatility': round(metrics.volatility, 2),
                    'calmar_ratio': round(metrics.calmar_ratio, 2)
                },
                'trading_statistics': {
                    'total_trades': metrics.total_trades,
                    'profitable_trades': metrics.profitable_trades,
                    'losing_trades': metrics.losing_trades,
                    'avg_hold_time_days': round(metrics.avg_hold_time_days, 1)
                },
                'current_portfolio': {
                    'total_positions': portfolio_performance.get('total_positions', 0),
                    'total_market_value': portfolio_performance.get('total_market_value', 0),
                    'total_unrealized_pnl': portfolio_performance.get('total_unrealized_pnl', 0),
                    'average_return_pct': portfolio_performance.get('average_return_pct', 0),
                    'risk_distribution': portfolio_performance.get('risk_distribution', {}),
                    'positions_needing_attention': len(portfolio_performance.get('positions_needing_attention', []))
                },
                'performance_analysis': self._analyze_performance(metrics),
                'recommendations': self._generate_performance_recommendations(metrics, portfolio_performance)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {str(e)}")
            return {'error': f"Performance report generation failed: {str(e)}"}
    
    def track_daily_performance(self) -> Dict:
        """
        Track and log daily portfolio performance.
        
        Returns:
            Dictionary containing daily performance snapshot
        """
        try:
            today = datetime.now().date().isoformat()
            
            # Get current portfolio status
            portfolio_performance = self.position_manager.get_portfolio_performance()
            
            # Create daily snapshot
            snapshot = {
                'date': today,
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': portfolio_performance.get('total_market_value', 0),
                'unrealized_pnl': portfolio_performance.get('total_unrealized_pnl', 0),
                'total_positions': portfolio_performance.get('total_positions', 0),
                'avg_return_pct': portfolio_performance.get('average_return_pct', 0),
                'risk_distribution': portfolio_performance.get('risk_distribution', {}),
                'top_performer': portfolio_performance.get('top_performers', [{}])[0] if portfolio_performance.get('top_performers') else None,
                'worst_performer': portfolio_performance.get('worst_performers', [{}])[0] if portfolio_performance.get('worst_performers') else None
            }
            
            # Save to performance tracking file
            self._save_daily_snapshot(snapshot)
            
            self.logger.info(f"Daily performance tracked: {snapshot['portfolio_value']:.2f} portfolio value")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to track daily performance: {str(e)}")
            return {'error': f"Daily performance tracking failed: {str(e)}"}
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_return_pct=0,
            annualized_return_pct=0,
            sharpe_ratio=0,
            max_drawdown_pct=0,
            win_rate=0,
            avg_win_pct=0,
            avg_loss_pct=0,
            profit_factor=0,
            total_trades=0,
            profitable_trades=0,
            losing_trades=0,
            avg_hold_time_days=0,
            volatility=0,
            calmar_ratio=0
        )
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(returns) < 2:
            return 0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        return (variance ** 0.5) * (365 ** 0.5)  # Annualized volatility
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0
        
        cumulative = [1]
        for r in returns:
            cumulative.append(cumulative[-1] * (1 + r / 100))
        
        peak = cumulative[0]
        max_dd = 0
        
        for value in cumulative:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _analyze_performance(self, metrics: PerformanceMetrics) -> Dict:
        """Analyze performance metrics and provide insights."""
        analysis = {
            'overall_rating': 'Poor',
            'strengths': [],
            'weaknesses': [],
            'key_insights': []
        }
        
        # Overall rating based on multiple factors
        score = 0
        
        if metrics.total_return_pct > 10:
            score += 2
            analysis['strengths'].append("Strong positive returns")
        elif metrics.total_return_pct > 0:
            score += 1
            analysis['strengths'].append("Positive returns")
        else:
            analysis['weaknesses'].append("Negative returns")
        
        if metrics.sharpe_ratio > 1.5:
            score += 2
            analysis['strengths'].append("Excellent risk-adjusted returns")
        elif metrics.sharpe_ratio > 1:
            score += 1
            analysis['strengths'].append("Good risk-adjusted returns")
        else:
            analysis['weaknesses'].append("Poor risk-adjusted returns")
        
        if metrics.win_rate > 60:
            score += 2
            analysis['strengths'].append("High win rate")
        elif metrics.win_rate > 50:
            score += 1
            analysis['strengths'].append("Decent win rate")
        else:
            analysis['weaknesses'].append("Low win rate")
        
        if metrics.max_drawdown_pct < 10:
            score += 1
            analysis['strengths'].append("Low drawdown")
        else:
            analysis['weaknesses'].append("High drawdown")
        
        if metrics.profit_factor > 2:
            score += 1
            analysis['strengths'].append("Strong profit factor")
        elif metrics.profit_factor < 1:
            analysis['weaknesses'].append("Poor profit factor")
        
        # Determine overall rating
        if score >= 6:
            analysis['overall_rating'] = 'Excellent'
        elif score >= 4:
            analysis['overall_rating'] = 'Good'
        elif score >= 2:
            analysis['overall_rating'] = 'Fair'
        
        # Generate insights
        if metrics.avg_win_pct > abs(metrics.avg_loss_pct):
            analysis['key_insights'].append("Winners are larger than losers on average")
        else:
            analysis['key_insights'].append("Need to improve win/loss ratio")
        
        if metrics.avg_hold_time_days < 7:
            analysis['key_insights'].append("Short-term trading strategy")
        elif metrics.avg_hold_time_days > 30:
            analysis['key_insights'].append("Long-term holding strategy")
        
        return analysis
    
    def _generate_performance_recommendations(self, metrics: PerformanceMetrics, portfolio: Dict) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []
        
        if metrics.win_rate < 50:
            recommendations.append("Consider improving trade selection criteria")
        
        if metrics.sharpe_ratio < 1:
            recommendations.append("Focus on risk management to improve risk-adjusted returns")
        
        if metrics.max_drawdown_pct > 20:
            recommendations.append("Implement stronger drawdown protection measures")
        
        if metrics.profit_factor < 1.5:
            recommendations.append("Review trade exit strategies to improve profit factor")
        
        if len(portfolio.get('positions_needing_attention', [])) > 3:
            recommendations.append("Review and potentially close high-risk positions")
        
        risk_dist = portfolio.get('risk_distribution', {})
        high_risk = risk_dist.get('high', 0) + risk_dist.get('critical', 0)
        if high_risk > portfolio.get('total_positions', 1) * 0.3:
            recommendations.append("Reduce overall portfolio risk exposure")
        
        return recommendations
    
    def _save_daily_snapshot(self, snapshot: Dict):
        """Save daily performance snapshot to file."""
        try:
            import os
            performance_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'performance')
            os.makedirs(performance_dir, exist_ok=True)
            
            filename = os.path.join(performance_dir, 'daily_performance.json')
            
            # Load existing data
            daily_data = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        daily_data = json.load(f)
                except:
                    daily_data = []
            
            # Add new snapshot
            daily_data.append(snapshot)
            
            # Keep only last 365 days
            if len(daily_data) > 365:
                daily_data = daily_data[-365:]
            
            # Save back to file
            with open(filename, 'w') as f:
                json.dump(daily_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save daily snapshot: {str(e)}")