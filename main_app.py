#!/usr/bin/env python3
"""
Comprehensive Main Entry Point for Crypto Trading Bot Application
===============================================================

This is the primary entry point for the enhanced crypto trading bot system.
It provides multiple execution modes including:

1. Interactive Dashboard Mode - Real-time portfolio monitoring
2. Automated Trading Mode - Continuous LSTM-based trading
3. Analysis Mode - Performance analytics and recommendations
4. Backtest Mode - Historical strategy validation
5. Data Collection Mode - Market data gathering
6. Setup Mode - System initialization and configuration
"""

import os
import sys
import argparse
import time
import signal
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core system imports
from config.config import AppConfig
from utils.logging_utils import setup_default_logging, get_logger, PerformanceLogger

# Trading system imports
from trading.trading_controller import TradingController
from trading.performance_analytics import PerformanceAnalytics
from exchanges.alpaca_client import AlpacaCryptoTrading
from data.crypto_data_client import CryptoMarketDataClient

# Model and analysis imports
from models.lstm_model_v2 import AggressiveCryptoSignalGenerator
from models.lstm_backtest import Backtester

# LLM integration
from llm.llm_client import LLMClient


class CryptoTradingBotApp:
    """
    Main application class for the crypto trading bot.
    Handles different execution modes and coordinates all system components.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the trading bot application."""
        
        # Set up logging first
        setup_default_logging()
        self.logger = get_logger(__name__)
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 CRYPTO TRADING BOT APPLICATION STARTING")
        self.logger.info("=" * 80)
        
        # Initialize configuration
        self.config = config or AppConfig()
        self.running = False
        self.mode = None
        
        # Initialize core components
        self.trading_controller = None
        self.signal_generator = None
        self.data_client = None
        self.alpaca_client = None
        self.llm_client = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Application initialization complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def initialize_components(self) -> bool:
        """Initialize all system components and validate connections."""
        
        self.logger.info("Initializing system components...")
        
        try:
            with PerformanceLogger(self.logger, "component_initialization"):
                
                # Initialize data client
                self.data_client = CryptoMarketDataClient()
                self.logger.info("✓ Data client initialized")
                
                # Initialize Alpaca client
                alpaca_config = self.config.get_alpaca_config()
                if alpaca_config['api_key'] and alpaca_config['api_secret']:
                    self.alpaca_client = AlpacaCryptoTrading(
                        api_key=alpaca_config['api_key'],
                        api_secret=alpaca_config['api_secret'],
                        base_url=alpaca_config['base_url']
                    )
                    self.logger.info("✓ Alpaca client initialized")
                else:
                    self.logger.warning("⚠️ Alpaca credentials not found - trading disabled")
                
                # Initialize trading controller
                if self.alpaca_client:
                    self.trading_controller = TradingController(self.config)
                    self.logger.info("✓ Trading controller initialized")
                
                # Initialize LSTM signal generator
                self.signal_generator = AggressiveCryptoSignalGenerator(ticker=self.config.ticker_slash)
                self.logger.info("✓ LSTM signal generator initialized")
                
                # Initialize LLM client
                llm_config = self.config.get_llm_config()
                if llm_config['api_key']:
                    self.llm_client = LLMClient()
                    self.logger.info("✓ LLM client initialized")
                else:
                    self.logger.warning("⚠️ LLM API key not found - AI features disabled")
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {str(e)}")
            return False
    
    def run_interactive_dashboard(self):
        """Run interactive dashboard mode with real-time updates."""
        
        self.logger.info("🖥️ Starting Interactive Dashboard Mode")
        self.mode = "dashboard"
        self.running = True
        
        if not self.trading_controller:
            print("❌ Trading controller not available. Please check Alpaca credentials.")
            return
        
        try:
            while self.running:
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("=" * 80)
                print("🚀 CRYPTO TRADING BOT - LIVE DASHBOARD")
                print("=" * 80)
                print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print()
                
                # Get portfolio dashboard
                dashboard = self.trading_controller.get_portfolio_dashboard()
                self._display_dashboard(dashboard)
                
                # Update every 30 seconds
                for i in range(30):
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Dashboard mode interrupted by user")
        except Exception as e:
            self.logger.error(f"Dashboard mode error: {str(e)}")
        finally:
            self.running = False
    
    def run_automated_trading(self, duration_hours: int = 24):
        """Run automated trading mode with LSTM signals."""
        
        self.logger.info(f"🤖 Starting Automated Trading Mode for {duration_hours} hours")
        self.mode = "trading"
        self.running = True
        
        if not self.trading_controller or not self.signal_generator:
            print("❌ Trading components not available. Please check configuration.")
            return
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        try:
            while self.running and datetime.now() < end_time:
                
                # Generate trading signal
                signal_data = self._generate_trading_signal()
                
                if signal_data:
                    self.logger.info(f"Generated signal: {signal_data}")
                    
                    # Process signal with LLM if available
                    if self.llm_client:
                        enhanced_signal = self._enhance_signal_with_llm(signal_data)
                        signal_data.update(enhanced_signal)
                    
                    # Execute trade if conditions are met
                    self._execute_trade_signal(signal_data)
                
                # Run daily analysis
                if datetime.now().hour == 0 and datetime.now().minute < 5:
                    self.logger.info("Running daily analysis...")
                    analysis_results = self.trading_controller.run_daily_analysis()
                    self._process_daily_analysis(analysis_results)
                
                # Wait before next iteration (15 minutes)
                for i in range(900):  # 15 minutes
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Automated trading interrupted by user")
        except Exception as e:
            self.logger.error(f"Automated trading error: {str(e)}")
        finally:
            self.running = False
    
    def run_analysis_mode(self, period_days: int = 30):
        """Run comprehensive analysis and generate reports."""
        
        self.logger.info(f"📊 Running Analysis Mode for {period_days} days")
        
        if not self.trading_controller:
            print("❌ Trading controller not available. Please check Alpaca credentials.")
            return
        
        try:
            # Generate comprehensive analysis
            analysis_results = self.trading_controller.run_daily_analysis()
            
            # Generate performance report
            performance_report = self.trading_controller.performance_analytics.generate_performance_report(period_days)
            
            # Display results
            self._display_analysis_results(analysis_results, performance_report)
            
            # Save reports
            self._save_analysis_reports(analysis_results, performance_report)
            
        except Exception as e:
            self.logger.error(f"Analysis mode error: {str(e)}")
    
    def run_backtest_mode(self, hours: int = 720, initial_balance: float = 100000):
        """Run backtesting mode to validate strategies."""
        
        self.logger.info(f"🔙 Running Backtest Mode for {hours} hours")
        
        try:
            # Generate signals for backtest period
            signal_generator = AggressiveCryptoSignalGenerator(ticker=self.config.ticker_slash)
            
            # Initialize backtester with the signal generator
            backtester = Backtester(
                signal_generator=signal_generator,
                hours=hours,
                initial_balance=initial_balance,
                fee=0.001  # 0.1% trading fee
            )
            
            # Run backtest
            results = backtester.run(verbose=True)
            
            # Display results
            self._display_backtest_results(results)
            
        except Exception as e:
            self.logger.error(f"Backtest mode error: {str(e)}")
            print(f"❌ Backtest failed: {str(e)}")
    
    def run_data_collection_mode(self, duration_hours: int = 1):
        """Run data collection mode to gather market data."""
        
        self.logger.info(f"📈 Running Data Collection Mode for {duration_hours} hours")
        self.mode = "data_collection"
        self.running = True
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        try:
            while self.running and datetime.now() < end_time:
                
                # Collect current market data
                market_data = self._collect_market_data()
                
                if market_data:
                    self.logger.info(f"Collected market data: {len(market_data)} data points")
                    
                    # Save to database/file
                    self._save_market_data(market_data)
                
                # Wait 5 minutes before next collection
                for i in range(300):
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Data collection interrupted by user")
        except Exception as e:
            self.logger.error(f"Data collection error: {str(e)}")
        finally:
            self.running = False
    
    def run_setup_mode(self):
        """Run setup mode to initialize and configure the system."""
        
        self.logger.info("⚙️ Running Setup Mode")
        
        print("🔧 CRYPTO TRADING BOT SETUP")
        print("=" * 50)
        
        # Check system requirements
        self._check_system_requirements()
        
        # Validate configuration
        self._validate_configuration()
        
        # Test connections
        self._test_connections()
        
        # Initialize directories
        self._initialize_directories()
        
        print("\n✅ Setup complete! You can now run the trading bot.")
    
    def _display_dashboard(self, dashboard: Dict):
        """Display formatted dashboard information."""
        
        if 'error' in dashboard:
            print(f"❌ Dashboard Error: {dashboard['error']}")
            return
        
        portfolio = dashboard.get('portfolio_overview', {})
        
        # Portfolio overview
        print("💰 PORTFOLIO OVERVIEW")
        print("-" * 30)
        print(f"Total Value: ${portfolio.get('total_market_value', 0):,.2f}")
        print(f"Unrealized P&L: ${portfolio.get('total_unrealized_pnl', 0):,.2f}")
        print(f"Total Positions: {portfolio.get('total_positions', 0)}")
        print(f"Average Return: {portfolio.get('average_return_pct', 0):+.2f}%")
        print()
        
        # Risk analysis
        risk_analysis = dashboard.get('risk_analysis', {})
        print("🚨 RISK DISTRIBUTION")
        print("-" * 30)
        print(f"Low Risk: {risk_analysis.get('low_risk_pct', 0):.1f}%")
        print(f"Medium Risk: {risk_analysis.get('medium_risk_pct', 0):.1f}%")
        print(f"High Risk: {risk_analysis.get('high_risk_pct', 0):.1f}%")
        print(f"Critical Risk: {risk_analysis.get('critical_risk_pct', 0):.1f}%")
        print()
        
        # Individual positions
        positions = dashboard.get('position_metrics', [])
        if positions:
            print("📈 INDIVIDUAL POSITIONS")
            print("-" * 30)
            for pos in positions[:5]:  # Show top 5
                symbol = pos.get('symbol', 'Unknown')
                value = pos.get('market_value', 0)
                pnl = pos.get('unrealized_pnl', 0)
                return_pct = pos.get('unrealized_pnl_pct', 0)
                risk = pos.get('risk_level', 'unknown')
                
                print(f"{symbol}: ${value:,.2f} | P&L: ${pnl:,.2f} ({return_pct:+.2f}%) | Risk: {risk}")
        
        # Recommendations summary
        rec_summary = dashboard.get('recommendations_summary', {})
        total_recs = sum(rec_summary.values())
        if total_recs > 0:
            print(f"\n⚡ RECOMMENDATIONS: {total_recs} total")
            print(f"   Urgent: {rec_summary.get('urgent_count', 0)}")
            print(f"   High: {rec_summary.get('high_count', 0)}")
            print(f"   Medium: {rec_summary.get('medium_count', 0)}")
            print(f"   Low: {rec_summary.get('low_count', 0)}")
        
        print("\n" + "=" * 80)
        print("Press Ctrl+C to exit dashboard mode")
    
    def _generate_trading_signal(self) -> Optional[Dict]:
        """Generate trading signal using LSTM model."""
        
        try:
            # Generate signal using the new aggressive method
            signal_result = self.signal_generator.generate_aggressive_signals(
                hours=self.config.lookback,
                retrain_threshold=6
            )
            
            if 'error' in signal_result:
                self.logger.warning(f"Signal generation error: {signal_result['error']}")
                return None
            
            if signal_result and signal_result.get('signal') != 0:
                return {
                    'symbol': self.config.ticker_slash,
                    'signal': 'BUY' if signal_result['signal'] > 0 else 'SELL',
                    'confidence': signal_result['confidence'],
                    'current_price': signal_result['current_price'],
                    'predicted_price': signal_result.get('predicted_price', signal_result['current_price']),
                    'timestamp': signal_result['timestamp'],
                    'indicators': {
                        'momentum_score': signal_result.get('momentum_score', 0),
                        'volatility_breakout': signal_result.get('volatility_breakout', False),
                        'volume_surge': signal_result.get('volume_surge', 1.0),
                        'rsi_fast': signal_result.get('rsi_fast', 50),
                        'risk_multiplier': signal_result.get('risk_multiplier', 1.0)
                    },
                    'trade_recommendation': signal_result.get('trade_recommendation', {})
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {str(e)}")
            return None
    
    def _enhance_signal_with_llm(self, signal_data: Dict) -> Dict:
        """Enhance trading signal with LLM analysis."""
        
        try:
            prompt = f"""
            Analyze this trading signal for {signal_data['symbol']}:
            
            Signal: {signal_data['signal']}
            Confidence: {signal_data['confidence']}
            Current Price: ${signal_data['current_price']:.2f}
            
            Market Context:
            - Current market conditions
            - Recent news sentiment
            - Technical indicators
            
            Provide:
            1. Signal validation (agree/disagree)
            2. Risk assessment (1-10)
            3. Position size recommendation (%)
            4. Stop loss suggestion
            5. Take profit targets
            """
            
            response = self.llm_client.query(prompt, provider="openai")
            
            # Parse LLM response (simplified)
            return {
                'llm_analysis': response,
                'llm_enhanced': True,
                'llm_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"LLM enhancement error: {str(e)}")
            return {'llm_enhanced': False}
    
    def _execute_trade_signal(self, signal_data: Dict):
        """Execute trade based on signal data."""
        
        try:
            if not self.alpaca_client:
                self.logger.warning("No trading client available")
                return
            
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Only execute high-confidence signals
            if confidence < 0.7:
                self.logger.info(f"Signal confidence too low: {confidence}")
                return
            
            # Calculate position size based on risk parameters
            position_size = self.config.risk_params['position_size']
            
            if signal == 'BUY':
                result = self.alpaca_client.place_order(
                    symbol=symbol.replace('/', ''),
                    qty=position_size,
                    side='buy',
                    type='market'
                )
                
                self.logger.info(f"Buy order executed: {result}")
                
            elif signal == 'SELL':
                # Close existing position
                result = self.alpaca_client.place_order(
                    symbol=symbol.replace('/', ''),
                    qty=position_size,
                    side='sell',
                    type='market'
                )
                
                self.logger.info(f"Sell order executed: {result}")
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")
    
    def _collect_market_data(self) -> Optional[Dict]:
        """Collect current market data."""
        
        try:
            # Get current price data
            current_bars = self.data_client.get_historical_bars(
                self.config.ticker_slash, 
                hours=1
            )
            
            if current_bars is not None and len(current_bars) > 0:
                latest = current_bars.iloc[-1]
                
                return {
                    'symbol': self.config.ticker_slash,
                    'timestamp': datetime.now().isoformat(),
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': float(latest['volume'])
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Market data collection error: {str(e)}")
            return None
    
    def _save_market_data(self, market_data: Dict):
        """Save market data to storage."""
        
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data', 'market_data')
            os.makedirs(data_dir, exist_ok=True)
            
            filename = os.path.join(data_dir, 'market_data.json')
            
            # Load existing data
            all_data = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        all_data = json.load(f)
                except:
                    all_data = []
            
            # Add new data
            all_data.append(market_data)
            
            # Keep only last 10000 records
            if len(all_data) > 10000:
                all_data = all_data[-10000:]
            
            # Save back to file
            with open(filename, 'w') as f:
                json.dump(all_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Market data save error: {str(e)}")
    
    def _check_system_requirements(self):
        """Check system requirements and dependencies."""
        
        print("🔍 Checking system requirements...")
        
        requirements = [
            ('Python version', sys.version_info >= (3, 8)),
            ('Config module', self._check_import('config.config')),
            ('Trading module', self._check_import('trading.trading_controller')),
            ('Data client', self._check_import('data.crypto_data_client')),
            ('LSTM model', self._check_import('models.lstm_model_v2')),
            ('Alpaca client', self._check_import('exchanges.alpaca_client')),
        ]
        
        all_good = True
        for name, check in requirements:
            status = "✅" if check else "❌"
            print(f"   {status} {name}")
            if not check:
                all_good = False
        
        if not all_good:
            print("\n⚠️ Some requirements are missing. Please install dependencies.")
            return False
        
        print("✅ All system requirements met")
        return True
    
    def _check_import(self, module_name: str) -> bool:
        """Check if a module can be imported."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _validate_configuration(self):
        """Validate configuration settings."""
        
        print("⚙️ Validating configuration...")
        
        # Check API keys
        alpaca_config = self.config.get_alpaca_config()
        llm_config = self.config.get_llm_config()
        
        configs = [
            ('Alpaca API Key', bool(alpaca_config.get('api_key'))),
            ('Alpaca Secret', bool(alpaca_config.get('api_secret'))),
            ('OpenAI API Key', bool(llm_config.get('api_key'))),
        ]
        
        for name, valid in configs:
            status = "✅" if valid else "⚠️"
            print(f"   {status} {name}")
        
        print("✅ Configuration validation complete")
    
    def _test_connections(self):
        """Test connections to external services."""
        
        print("🌐 Testing connections...")
        
        # Test data client
        try:
            test_data = self.data_client.get_historical_bars('BTC/USD', hours=1)
            data_status = "✅" if test_data is not None else "❌"
        except:
            data_status = "❌"
        
        print(f"   {data_status} Market data connection")
        
        # Test Alpaca connection
        if self.alpaca_client:
            try:
                account_info = self.alpaca_client.get_account()
                alpaca_status = "✅" if 'id' in account_info else "❌"
            except:
                alpaca_status = "❌"
        else:
            alpaca_status = "⚠️"
        
        print(f"   {alpaca_status} Alpaca trading connection")
        
        print("✅ Connection testing complete")
    
    def _initialize_directories(self):
        """Initialize required directories."""
        
        print("📁 Initializing directories...")
        
        directories = [
            'logs',
            'data/market_data',
            'data/performance',
            'models/saved',
            'reports'
        ]
        
        for directory in directories:
            dir_path = os.path.join(os.path.dirname(__file__), directory)
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ {directory}")
        
        print("✅ Directory initialization complete")
    
    def _display_analysis_results(self, analysis_results: Dict, performance_report: Dict):
        """Display analysis results."""
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE ANALYSIS RESULTS")
        print("=" * 80)
        
        # Summary
        summary = analysis_results.get('summary', {})
        print(f"Analysis Status: {summary.get('analysis_status', 'unknown')}")
        print(f"Total Positions: {summary.get('total_positions', 0)}")
        print(f"Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}")
        print(f"Unrealized P&L: ${summary.get('unrealized_pnl', 0):,.2f}")
        print(f"Total Recommendations: {summary.get('total_recommendations', 0)}")
        print(f"Urgent Actions Needed: {summary.get('urgent_actions_needed', 0)}")
        
        # Performance metrics
        if 'performance_metrics' in performance_report:
            metrics = performance_report['performance_metrics']
            print(f"\n📈 PERFORMANCE METRICS")
            print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        
        # Key messages
        key_messages = summary.get('key_messages', [])
        if key_messages:
            print(f"\n⚡ KEY INSIGHTS")
            for message in key_messages:
                print(f"   • {message}")
        
        # Recommendations
        if 'recommendations' in performance_report:
            recommendations = performance_report['recommendations']
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS")
                for rec in recommendations:
                    print(f"   • {rec}")
    
    def _display_backtest_results(self, results: Dict):
        """Display backtest results."""
        
        print("\n" + "=" * 80)
        print("🔙 BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"Initial Balance: ${results.get('initial_balance', 0):,.2f}")
        print(f"Final Balance: ${results.get('final_balance', 0):,.2f}")
        print(f"Total Return: {results.get('total_return_pct', 0):.2f}%")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Winning Trades: {results.get('winning_trades', 0)}")
        print(f"Losing Trades: {results.get('losing_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    
    def _save_analysis_reports(self, analysis_results: Dict, performance_report: Dict):
        """Save analysis reports to files."""
        
        try:
            reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save analysis results
            analysis_file = os.path.join(reports_dir, f'analysis_{timestamp}.json')
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Save performance report
            performance_file = os.path.join(reports_dir, f'performance_{timestamp}.json')
            with open(performance_file, 'w') as f:
                json.dump(performance_report, f, indent=2)
            
            self.logger.info(f"Reports saved: {analysis_file}, {performance_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save reports: {str(e)}")
    
    def _process_daily_analysis(self, analysis_results: Dict):
        """Process daily analysis results."""
        
        try:
            # Log key metrics
            summary = analysis_results.get('summary', {})
            self.logger.info(f"Daily analysis: {summary.get('total_positions', 0)} positions, "
                           f"${summary.get('portfolio_value', 0):,.2f} value, "
                           f"{summary.get('urgent_actions_needed', 0)} urgent actions")
            
            # Handle urgent recommendations
            urgent_count = summary.get('urgent_actions_needed', 0)
            if urgent_count > 0:
                self.logger.warning(f"⚠️ {urgent_count} urgent recommendations require attention!")
                
                # Could implement automated responses here
                # For now, just log for manual review
                
        except Exception as e:
            self.logger.error(f"Daily analysis processing error: {str(e)}")


def main():
    """Main entry point with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot - Comprehensive Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_app.py --mode dashboard                    # Interactive dashboard
  python main_app.py --mode trading --duration 24       # Automated trading for 24 hours
  python main_app.py --mode analysis --period 30        # 30-day performance analysis
  python main_app.py --mode backtest --hours 720        # Backtest for 720 hours
  python main_app.py --mode data --duration 1           # Data collection for 1 hour
  python main_app.py --mode setup                       # System setup and validation
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['dashboard', 'trading', 'analysis', 'backtest', 'data', 'setup'],
        default='dashboard',
        help='Execution mode (default: dashboard)'
    )
    
    parser.add_argument(
        '--ticker',
        default='BTC',
        help='Crypto ticker symbol (default: BTC)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=24,
        help='Duration in hours for trading/data modes (default: 24)'
    )
    
    parser.add_argument(
        '--period',
        type=int,
        default=30,
        help='Analysis period in days (default: 30)'
    )
    
    parser.add_argument(
        '--hours',
        type=int,
        default=720,
        help='Backtest hours (default: 720)'
    )
    
    parser.add_argument(
        '--balance',
        type=float,
        default=100000,
        help='Initial balance for backtest (default: 100000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = AppConfig(ticker=args.ticker)
        if args.debug:
            config.log_level = "DEBUG"
        
        # Initialize application
        app = CryptoTradingBotApp(config)
        
        print(f"🚀 Starting Crypto Trading Bot in {args.mode.upper()} mode")
        print(f"📊 Ticker: {args.ticker}")
        
        # Initialize components
        if not app.initialize_components():
            print("❌ Component initialization failed. Exiting.")
            return 1
        
        # Run selected mode
        if args.mode == 'dashboard':
            app.run_interactive_dashboard()
            
        elif args.mode == 'trading':
            app.run_automated_trading(duration_hours=args.duration)
            
        elif args.mode == 'analysis':
            app.run_analysis_mode(period_days=args.period)
            
        elif args.mode == 'backtest':
            app.run_backtest_mode(hours=args.hours, initial_balance=args.balance)
            
        elif args.mode == 'data':
            app.run_data_collection_mode(duration_hours=args.duration)
            
        elif args.mode == 'setup':
            app.run_setup_mode()
        
        print("✅ Application completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚡ Application interrupted by user")
        return 0
        
    except Exception as e:
        print(f"❌ Application error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
