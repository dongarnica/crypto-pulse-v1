"""
Trading module for LSTM-based automated crypto trading.

This module contains the TradingController class and trading functionality.
"""

from .trading_controller import TradingController
from .position_manager import PositionManager
from .performance_analytics import PerformanceAnalytics
from .recommendations import TradingRecommendationEngine, TradingRecommendation, Priority

__all__ = ['TradingController', 'PositionManager', 'PerformanceAnalytics', 
           'TradingRecommendationEngine', 'TradingRecommendation', 'Priority']
