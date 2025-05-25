#!/usr/bin/env python3
# Enhanced training optimization for aggressive crypto trading LSTM model

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
import traceback
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.lstm_model_v2 import AggressiveCryptoSignalGenerator
from utils.logging_utils import setup_logging

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger('app_lstmv2_optimized')

# Define fallback tickers to try if the requested ticker fails
FALLBACK_TICKERS = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD']

# Training optimization configurations for different volatility regimes
TRAINING_CONFIGS = {
    'high_volatility': {
        'lookback_periods': [15, 20, 30],  # Shorter for rapid market changes
        'hours_data': [48, 72, 168],       # 2 days to 1 week
        'retrain_threshold': 2,            # Retrain more frequently
        'validation_split': 0.25,          # More validation data
        'early_stopping_patience': 15,    # Less patience for quick adaptation
        'learning_rate_schedule': 'aggressive',
        'dropout_rate': 0.3,
        'batch_size': 32
    },
    'medium_volatility': {
        'lookback_periods': [20, 30, 45],
        'hours_data': [72, 168, 336],      # 3 days to 2 weeks  
        'retrain_threshold': 4,
        'validation_split': 0.2,
        'early_stopping_patience': 20,
        'learning_rate_schedule': 'moderate',
        'dropout_rate': 0.25,
        'batch_size': 64
    },
    'low_volatility': {
        'lookback_periods': [30, 45, 60],  # Longer for stable patterns
        'hours_data': [168, 336, 720],     # 1 week to 1 month
        'retrain_threshold': 6,            # Less frequent retraining
        'validation_split': 0.15,
        'early_stopping_patience': 30,    # More patience for convergence
        'learning_rate_schedule': 'conservative',
        'dropout_rate': 0.2,
        'batch_size': 128
    }
}

class OptimalTrainingManager:
    """Manages optimal training parameters based on market conditions."""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.training_history = {}
        self.performance_cache = {}
        
    def detect_market_volatility(self, ticker: str, hours: int = 168) -> str:
        """Detect current market volatility regime."""
        try:
            # This would connect to your data source to calculate volatility
            # For now, return a placeholder - you'll need to implement actual volatility calculation
            temp_model = AggressiveCryptoSignalGenerator(ticker=ticker, model_dir=self.model_dir)
            
            # Fetch recent data and calculate volatility metrics
            # You'll need to implement this based on your data source
            volatility_score = self._calculate_volatility_score(temp_model, hours)
            
            if volatility_score > 0.7:
                return 'high_volatility'
            elif volatility_score > 0.4:
                return 'medium_volatility'
            else:
                return 'low_volatility'
                
        except Exception as e:
            logger.warning(f"Could not detect volatility, defaulting to medium: {e}")
            return 'medium_volatility'
    
    def _calculate_volatility_score(self, model, hours: int) -> float:
        """Calculate a normalized volatility score (0-1)."""
        # Placeholder implementation - you'll need to implement based on your data source
        # This should calculate rolling volatility over different timeframes
        return 0.5  # Default to medium volatility
    
    def find_optimal_parameters(self, ticker: str, volatility_regime: str = None) -> Dict:
        """Find optimal training parameters through grid search."""
        if volatility_regime is None:
            volatility_regime = self.detect_market_volatility(ticker)
        
        config = TRAINING_CONFIGS[volatility_regime]
        logger.info(f"Using {volatility_regime} configuration for {ticker}")
        
        best_params = None
        best_score = -float('inf')
        
        # Grid search over key parameters
        for lookback in config['lookback_periods']:
            for hours in config['hours_data']:
                # Skip if hours <= lookback (not enough data)
                if hours <= lookback * 2:
                    continue
                    
                try:
                    score = self._evaluate_parameter_combination(
                        ticker, lookback, hours, config
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'lookback': lookback,
                            'hours': hours,
                            'volatility_regime': volatility_regime,
                            'config': config,
                            'score': score
                        }
                        
                    logger.info(f"Tested lookback={lookback}, hours={hours}, score={score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate lookback={lookback}, hours={hours}: {e}")
                    continue
        
        if best_params is None:
            # Fallback to default medium volatility params
            config = TRAINING_CONFIGS['medium_volatility']
            best_params = {
                'lookback': config['lookback_periods'][1],  # Middle value
                'hours': config['hours_data'][1],
                'volatility_regime': 'medium_volatility',
                'config': config,
                'score': 0.0
            }
            logger.warning("Using fallback parameters due to evaluation failures")
        
        logger.info(f"Optimal parameters found: lookback={best_params['lookback']}, "
                   f"hours={best_params['hours']}, score={best_params['score']:.4f}")
        
        return best_params
    
    def _evaluate_parameter_combination(self, ticker: str, lookback: int, 
                                      hours: int, config: Dict) -> float:
        """Evaluate a specific parameter combination using walk-forward validation."""
        try:
            model = AggressiveCryptoSignalGenerator(
                ticker=ticker,
                model_dir=self.model_dir,
                lookback=lookback
            )
            
            # Configure model with optimal parameters
            model.volatility_multiplier = 2.5  # More aggressive for crypto
            model.momentum_weight = 2.0
            model.risk_amplification = 2.2
            
            # Set training parameters from config
            if hasattr(model, 'set_training_params'):
                model.set_training_params(
                    validation_split=config['validation_split'],
                    early_stopping_patience=config['early_stopping_patience'],
                    learning_rate_schedule=config['learning_rate_schedule'],
                    dropout_rate=config['dropout_rate'],
                    batch_size=config['batch_size']
                )
            
            # Perform walk-forward validation
            scores = []
            validation_periods = 3  # Test on 3 different time periods
            
            for i in range(validation_periods):
                # Offset training data for walk-forward testing
                offset_hours = hours + (i * 24)  # 24-hour offset for each test
                
                try:
                    signal_data = model.generate_aggressive_signals(
                        hours=offset_hours, 
                        retrain_threshold=config['retrain_threshold']
                    )
                    
                    if 'error' not in signal_data:
                        # Calculate score based on confidence and signal strength
                        confidence = signal_data.get('confidence', 0)
                        signal_strength = abs(signal_data.get('signal', 0))
                        
                        # Penalty for neutral signals in aggressive trading
                        if signal_data.get('signal', 0) == 0:
                            confidence *= 0.5
                        
                        score = confidence * signal_strength
                        scores.append(score)
                    else:
                        scores.append(0.0)  # Penalty for errors
                        
                except Exception as e:
                    logger.debug(f"Validation period {i} failed: {e}")
                    scores.append(0.0)
            
            # Return average score with penalty for inconsistency
            if scores:
                avg_score = np.mean(scores)
                consistency_penalty = 1.0 - (np.std(scores) / (avg_score + 1e-8))
                return avg_score * max(0.1, consistency_penalty)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Parameter evaluation failed: {e}")
            return 0.0
    
    def should_retrain(self, model, current_performance: Dict) -> bool:
        """Determine if model should be retrained based on performance degradation."""
        if not hasattr(model, 'ticker'):
            return True
            
        ticker = model.ticker
        
        # Check if we have historical performance data
        if ticker not in self.performance_cache:
            self.performance_cache[ticker] = []
            return True
        
        recent_performances = self.performance_cache[ticker]
        
        # Always retrain if we have less than 5 performance records
        if len(recent_performances) < 5:
            recent_performances.append(current_performance)
            return True
        
        # Calculate performance trend
        recent_confidences = [p.get('avg_confidence', 0) for p in recent_performances[-5:]]
        current_confidence = current_performance.get('avg_confidence', 0)
        
        # Retrain if confidence has dropped significantly
        avg_recent = np.mean(recent_confidences)
        confidence_drop = (avg_recent - current_confidence) / (avg_recent + 1e-8)
        
        # Store current performance
        recent_performances.append(current_performance)
        if len(recent_performances) > 10:  # Keep only last 10 records
            recent_performances.pop(0)
        
        # Retrain if confidence dropped by more than 15%
        return confidence_drop > 0.15

def parse_arguments():
    """Parse command line arguments with optimization options."""
    parser = argparse.ArgumentParser(
        description='Run the Optimized Aggressive LSTM v2 Crypto Signal Generator'
    )
    
    # Original arguments
    parser.add_argument('--ticker', type=str, default='BTC/USD',
                       help='Crypto ticker symbol (default: BTC/USD)')
    parser.add_argument('--volatility-multiplier', type=float, default=2.5,
                       help='Volatility multiplier for signals (default: 2.5)')
    parser.add_argument('--momentum-weight', type=float, default=2.0,
                       help='Momentum weight for signals (default: 2.0)')
    parser.add_argument('--risk-amplification', type=float, default=2.2,
                       help='Risk amplification factor (default: 2.2)')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Directory to store model files (default: models/data)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file to save signal results as JSON')
    parser.add_argument('--continuous', action='store_true',
                       help='Run in continuous mode with periodic signal generation')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval between signal generations in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--fallback', action='store_true',
                       help='Try fallback tickers if primary ticker fails')
    
    # New optimization arguments
    parser.add_argument('--auto-optimize', action='store_true',
                       help='Automatically find optimal training parameters')
    parser.add_argument('--force-volatility-regime', type=str, 
                       choices=['high_volatility', 'medium_volatility', 'low_volatility'],
                       help='Force a specific volatility regime instead of auto-detection')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='Minimum confidence threshold for signals (default: 0.6)')
    parser.add_argument('--adaptive-retraining', action='store_true',
                       help='Enable adaptive retraining based on performance')
    parser.add_argument('--max-training-time', type=int, default=300,
                       help='Maximum training time in seconds (default: 300)')
    
    return parser.parse_args()

def main():
    """Main function with optimization capabilities."""
    args = parse_arguments()
    
    # Set up logging level based on verbose flag
    if args.verbose:
        setup_logging(log_level="DEBUG")
        logger.setLevel(logging.DEBUG)
    
    model_dir = args.model_dir
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"🚀 OPTIMIZED AGGRESSIVE LSTM V2 CRYPTO SIGNAL GENERATOR")
    print(f"{'='*80}")
    
    # Initialize optimization manager
    optimizer = OptimalTrainingManager(model_dir)
    
    # Find optimal parameters if requested
    if args.auto_optimize:
        print(f"🔍 Finding optimal parameters for {args.ticker}...")
        optimal_params = optimizer.find_optimal_parameters(
            args.ticker, 
            args.force_volatility_regime
        )
        lookback = optimal_params['lookback']
        hours = optimal_params['hours']
        config = optimal_params['config']
        
        print(f"✅ Optimal parameters found:")
        print(f"   • Lookback: {lookback}")
        print(f"   • Hours: {hours}")
        print(f"   • Volatility regime: {optimal_params['volatility_regime']}")
        print(f"   • Optimization score: {optimal_params['score']:.4f}")
    else:
        # Use default or detect volatility regime
        regime = args.force_volatility_regime or optimizer.detect_market_volatility(args.ticker)
        config = TRAINING_CONFIGS[regime]
        lookback = config['lookback_periods'][1]  # Use middle value
        hours = config['hours_data'][1]
        print(f"📊 Using {regime} configuration")
    
    print(f"• Ticker: {args.ticker}")
    print(f"• Lookback period: {lookback}")
    print(f"• Historical data: {hours} hours")
    print(f"• Retrain threshold: {config['retrain_threshold']}")
    print(f"• Min confidence: {args.min_confidence}")
    print(f"{'='*80}")
    
    try:
        # Initialize model with optimal parameters
        model = AggressiveCryptoSignalGenerator(
            ticker=args.ticker,
            model_dir=model_dir,
            lookback=lookback
        )
        
        # Configure model with optimized settings
        model.volatility_multiplier = args.volatility_multiplier
        model.momentum_weight = args.momentum_weight
        model.risk_amplification = args.risk_amplification
        
        if args.continuous:
            run_optimized_continuous_mode(model, optimizer, args, hours, config)
        else:
            signal_data = generate_optimized_signal(model, optimizer, args, hours, config)
            print_enhanced_signal_summary(signal_data, model, args)
            
    except KeyboardInterrupt:
        print("\n🛑 Program terminated by user")
    except Exception as e:
        logger.exception(f"Error running optimized LSTM v2 model: {e}")
        print(f"❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    
    return 0

def generate_optimized_signal(model, optimizer, args, hours, config):
    """Generate signal with optimization logic."""
    signal_data = None
    start_time = time.time()
    
    try:
        # Check if we should retrain
        should_retrain = True
        if args.adaptive_retraining:
            current_performance = model.get_performance_metrics()
            should_retrain = optimizer.should_retrain(model, current_performance)
            
            if should_retrain:
                print("🔄 Performance degradation detected, retraining model...")
            else:
                print("✅ Model performance is stable, skipping retraining")
        
        retrain_threshold = config['retrain_threshold'] if should_retrain else 0
        
        # Set maximum training time
        if hasattr(model, 'set_max_training_time'):
            model.set_max_training_time(args.max_training_time)
        
        signal_data = model.generate_aggressive_signals(
            hours=hours, 
            retrain_threshold=retrain_threshold
        )
        
        # Filter by confidence threshold
        if ('error' not in signal_data and 
            signal_data.get('confidence', 0) < args.min_confidence):
            
            print(f"⚠️  Signal confidence ({signal_data['confidence']:.2f}) "
                  f"below threshold ({args.min_confidence}), setting to neutral")
            signal_data['signal'] = 0
            signal_data['original_signal'] = signal_data.get('signal', 0)
            signal_data['confidence_filtered'] = True
    
    except Exception as e:
        logger.exception(f"Error generating optimized signal: {e}")
        signal_data = {"error": str(e), "signal": 0}
    
    execution_time = time.time() - start_time
    
    # Add optimization metadata
    if signal_data is None:
        signal_data = {"error": "Unknown error occurred", "signal": 0}
    
    signal_data['optimization_metadata'] = {
        'execution_time_seconds': round(execution_time, 2),
        'generated_at': datetime.now().isoformat(),
        'optimization_enabled': args.auto_optimize,
        'adaptive_retraining': args.adaptive_retraining,
        'min_confidence_threshold': args.min_confidence,
        'training_config_used': config,
        'parameters': {
            'ticker': args.ticker,
            'lookback': model.lookback if hasattr(model, 'lookback') else 'unknown',
            'hours': hours,
            'volatility_multiplier': args.volatility_multiplier,
            'momentum_weight': args.momentum_weight,
            'risk_amplification': args.risk_amplification
        }
    }
    
    return signal_data

def print_enhanced_signal_summary(signal_data, model, args):
    """Print enhanced signal summary with optimization info and positions."""
    if 'error' in signal_data:
        print(f"❌ Error generating signal: {signal_data['error']}")
        return
    
    signal = signal_data['signal']
    confidence = signal_data['confidence']
    
    # Enhanced signal display
    if signal == 1:
        signal_str = "🟢 STRONG LONG"
        signal_emoji = "📈"
    elif signal == -1:
        signal_str = "🔴 STRONG SHORT" 
        signal_emoji = "📉"
    else:
        signal_str = "⚪ NEUTRAL/HOLD"
        signal_emoji = "⏸️"
    
    # Confidence level assessment
    if confidence >= 0.8:
        confidence_level = "VERY HIGH"
    elif confidence >= 0.7:
        confidence_level = "HIGH"
    elif confidence >= 0.6:
        confidence_level = "MODERATE"
    else:
        confidence_level = "LOW"
    
    print(f"\n{signal_emoji} OPTIMIZED SIGNAL SUMMARY:")
    print(f"{'='*60}")
    print(f"Signal: {signal_str} ({signal})")
    print(f"Confidence: {confidence:.3f} ({confidence_level})")
    print(f"Current price: ${signal_data['current_price']:.2f}")
    print(f"Predicted price: ${signal_data['predicted_price']:.2f}")
    
    # Show if signal was filtered
    if signal_data.get('confidence_filtered'):
        print(f"⚠️  Original signal was filtered due to low confidence")
    
    # Current position information
    if 'trade_recommendation' in signal_data:
        rec = signal_data['trade_recommendation']
        current_position = rec.get('current_position')
        
        print(f"\n📊 POSITION INFORMATION:")
        print(f"{'='*60}")
        
        if current_position:
            pos_summary = model.data_client.get_position_summary(args.ticker)
            print(f"Current Position: {current_position['side'].upper()}")
            print(f"Entry Price: ${current_position['entry_price']:.2f}")
            print(f"Quantity: {current_position['quantity']:.4f}")
            print(f"Unrealized P&L: ${pos_summary['unrealized_pnl']:.2f} ({pos_summary['unrealized_pnl_pct']:+.2f}%)")
            
            if current_position.get('stop_loss'):
                print(f"Stop Loss: ${current_position['stop_loss']:.2f}")
            if current_position.get('take_profit'):
                print(f"Take Profit: ${current_position['take_profit']:.2f}")
        else:
            print("No current position")
        
        # Portfolio summary
        portfolio_summary = model.data_client.get_position_summary()
        if portfolio_summary.get('total_positions', 0) > 0:
            print(f"\n📈 PORTFOLIO SUMMARY:")
            print(f"{'='*60}")
            print(f"Total Positions: {portfolio_summary['total_positions']}")
            print(f"Long Positions: {portfolio_summary['long_positions']}")
            print(f"Short Positions: {portfolio_summary['short_positions']}")
            print(f"Total Exposure: ${portfolio_summary['total_exposure']:.2f}")
            print(f"Total Unrealized P&L: ${portfolio_summary['total_unrealized_pnl']:.2f}")
    
    # Optimization info
    if 'optimization_metadata' in signal_data:
        opt_meta = signal_data['optimization_metadata']
        print(f"\n⚙️  OPTIMIZATION INFO:")
        print(f"{'='*60}")
        print(f"Execution time: {opt_meta['execution_time_seconds']:.2f}s")
        print(f"Optimization enabled: {opt_meta['optimization_enabled']}")
        print(f"Adaptive retraining: {opt_meta['adaptive_retraining']}")
        print(f"Min confidence threshold: {opt_meta['min_confidence_threshold']}")
    
    # Enhanced trade recommendation with position considerations
    if 'trade_recommendation' in signal_data:
        rec = signal_data['trade_recommendation']
        print(f"\n💰 ENHANCED TRADE RECOMMENDATION:")
        print(f"{'='*60}")
        print(f"Action: {rec['action']}")
        
        if rec.get('position_size_pct', 0) > 0:
            print(f"Position Size: {rec['position_size_pct']}%")
        
        if rec.get('stop_loss'):
            print(f"Stop Loss: ${rec['stop_loss']:.2f}")
        if rec.get('take_profit'):
            print(f"Take Profit: ${rec['take_profit']:.2f}")
        if rec.get('risk_reward_ratio'):
            print(f"Risk/Reward Ratio: 1:{rec['risk_reward_ratio']:.2f}")
        
        print(f"Reasoning: {rec['reasoning']}")
        
        # Risk management info
        if 'risk_management' in rec:
            risk_mgmt = rec['risk_management']
            print(f"\n🛡️  RISK MANAGEMENT:")
            print(f"{'='*40}")
            print(f"Portfolio Heat: {risk_mgmt.get('portfolio_heat', 0):.1%}")
            
            if risk_mgmt.get('position_limit_reached'):
                print("⚠️  Position size limit reached")
            if risk_mgmt.get('correlation_warning'):
                print("⚠️  High correlation warning")
        
        # Portfolio impact
        if 'portfolio_impact' in rec:
            impact = rec['portfolio_impact']
            print(f"\n🎯 PORTFOLIO IMPACT:")
            print(f"{'='*40}")
            if 'current_exposure' in impact:
                print(f"Current Exposure: {impact['current_exposure']:.1%}")
                print(f"New Exposure: {impact['new_exposure']:.1%}")
                print(f"Available Exposure: {impact['available_exposure']:.1%}")

def run_optimized_continuous_mode(model, optimizer, args, hours, config):
    """Run continuous mode with optimization."""
    try:
        iteration = 1
        while True:
            print(f"\n{'='*60}")
            print(f"🔄 OPTIMIZATION ITERATION {iteration}")
            print(f"Time: {datetime.now().isoformat()}")
            print(f"{'='*60}")
            
            signal_data = generate_optimized_signal(model, optimizer, args, hours, config)
            print_enhanced_signal_summary(signal_data, model, args)
            
            iteration += 1
            print(f"\n⏱️  Next optimization in {args.interval} seconds...")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Optimized continuous mode stopped by user")

if __name__ == "__main__":
    sys.exit(main())