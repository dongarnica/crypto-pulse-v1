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

# Import display formatters
from utils.display_formatters import TradeFormatter, RecommendationFormatter, format_currency, format_percentage, create_progress_bar


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
        self.signal_generators = {}  # Dictionary to hold multiple signal generators
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
                
                # Initialize LSTM signal generators for active tickers
                active_tickers = self.config.get_active_tickers()
                self.logger.info(f"Initializing signal generators for {len(active_tickers)} tickers: {active_tickers}")
                
                for ticker in active_tickers:
                    try:
                        generator = AggressiveCryptoSignalGenerator(ticker=ticker)
                        self.signal_generators[ticker] = generator
                        self.logger.info(f"✓ Signal generator initialized for {ticker}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to initialize signal generator for {ticker}: {e}")
                
                # Set primary signal generator (backward compatibility)
                if self.config.ticker_slash in self.signal_generators:
                    self.signal_generator = self.signal_generators[self.config.ticker_slash]
                elif self.signal_generators:
                    self.signal_generator = list(self.signal_generators.values())[0]
                    self.logger.info(f"✓ Primary signal generator set to {list(self.signal_generators.keys())[0]}")
                else:
                    self.logger.warning("⚠️ No signal generators successfully initialized")
                
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
                
                # Display active tickers info
                active_tickers = self.config.get_active_tickers()
                if self.config.multi_ticker_params['enable_multi_ticker']:
                    print(f"📊 Multi-Ticker Mode: {len(active_tickers)} active tickers")
                    print(f"🎯 Active: {', '.join(active_tickers[:5])}" + ("..." if len(active_tickers) > 5 else ""))
                else:
                    print(f"📊 Single-Ticker Mode: {self.config.ticker_slash}")
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
                
                # Check if multi-ticker mode is enabled
                if self.config.multi_ticker_params['enable_multi_ticker']:
                    # Generate signals for all active tickers
                    signals = self._generate_multi_ticker_signals()
                    
                    if signals:
                        self.logger.info(f"Generated {len(signals)} signals across multiple tickers")
                        
                        # Process signals with LLM if available
                        enhanced_signals = []
                        for signal_data in signals:
                            if self.llm_client:
                                enhanced_signal = self._enhance_signal_with_llm(signal_data)
                                signal_data.update(enhanced_signal)
                            enhanced_signals.append(signal_data)
                        
                        # Execute trades with portfolio management
                        self._execute_multi_ticker_trades(enhanced_signals)
                    else:
                        self.logger.info("No signals generated for any active tickers")
                else:
                    # Single ticker mode (backward compatibility)
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
                
                # Collect market data for all active tickers
                active_tickers = self.config.get_active_tickers()
                all_market_data = []
                
                for ticker in active_tickers:
                    market_data = self._collect_market_data_for_ticker(ticker)
                    if market_data:
                        all_market_data.append(market_data)
                
                if all_market_data:
                    self.logger.info(f"Collected market data for {len(all_market_data)} tickers")
                    
                    # Save to database/file
                    self._save_market_data(all_market_data)
                
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
    
    def _collect_market_data_for_ticker(self, ticker: str) -> Optional[Dict]:
        """Collect current market data for a specific ticker."""
        
        try:
            # Get current price data
            current_bars = self.data_client.get_historical_bars(
                ticker, 
                hours=1
            )
            
            if current_bars is not None and len(current_bars) > 0:
                latest = current_bars.iloc[-1]
                
                return {
                    'symbol': ticker,
                    'timestamp': datetime.now().isoformat(),
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': float(latest['volume'])
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Market data collection error for {ticker}: {str(e)}")
            return None
    
    def _collect_market_data(self) -> Optional[Dict]:
        """Collect current market data for primary ticker (backward compatibility)."""
        return self._collect_market_data_for_ticker(self.config.ticker_slash)
    
    def _save_market_data(self, market_data):
        """Save market data to storage. Handles both single Dict and List[Dict]."""
        
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
            
            # Add new data (handle both single dict and list of dicts)
            if isinstance(market_data, list):
                all_data.extend(market_data)
            else:
                all_data.append(market_data)
            
            # Keep only last 10000 records
            if len(all_data) > 10000:
                all_data = all_data[-10000:]
            
            # Save back to file
            with open(filename, 'w') as f:
                json.dump(all_data, f, indent=2, cls=self.CustomJSONEncoder)
                
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
        print(f"Total Return: {results.get('total_return', 0)*100:.2f}%")
        print(f"Total Trades: {results.get('num_trades', 0)}")
        print(f"Winning Trades: {results.get('winning_trades', 0)}")
        print(f"Losing Trades: {results.get('losing_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0)*100:.2f}%")
        print(f"Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    
    def _save_analysis_reports(self, analysis_results: Dict, performance_report: Dict):
        """Save analysis reports to files."""
        
        def json_serializer(obj):
            """Custom JSON serializer for non-serializable objects."""
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return str(obj)
        
        try:
            reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save analysis results
            analysis_file = os.path.join(reports_dir, f'analysis_{timestamp}.json')
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=json_serializer)
            
            # Save performance report
            performance_file = os.path.join(reports_dir, f'performance_{timestamp}.json')
            with open(performance_file, 'w') as f:
                json.dump(performance_report, f, indent=2, default=json_serializer)
            
            self.logger.info(f"Reports saved: {analysis_file}, {performance_file}")
            print(f"📄 Reports saved to:")
            print(f"   📊 Analysis: {analysis_file}")
            print(f"   📈 Performance: {performance_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save reports: {str(e)}")
            print(f"❌ Failed to save reports: {str(e)}")
    
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
    
    def _generate_multi_ticker_signals(self) -> List[Dict]:
        """Generate trading signals for all active tickers."""
        
        signals = []
        active_tickers = self.config.get_active_tickers()
        
        self.logger.info(f"Generating signals for {len(active_tickers)} tickers")
        
        for ticker in active_tickers:
            if ticker not in self.signal_generators:
                self.logger.warning(f"No signal generator found for {ticker}")
                continue
                
            try:
                signal_result = self.signal_generators[ticker].generate_aggressive_signals(
                    hours=self.config.lookback,
                    retrain_threshold=6
                )
                
                if 'error' in signal_result:
                    self.logger.warning(f"Signal generation error for {ticker}: {signal_result['error']}")
                    continue
                
                if signal_result and signal_result.get('signal') != 0:
                    signal_data = {
                        'symbol': ticker,
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
                    signals.append(signal_data)
                    self.logger.info(f"Generated {signal_data['signal']} signal for {ticker} with confidence {signal_data['confidence']:.2f}")
                    
            except Exception as e:
                self.logger.error(f"Error generating signal for {ticker}: {str(e)}")
        
        return signals
    
    def _execute_multi_ticker_trades(self, signals: List[Dict]):
        """Execute trades for multiple ticker signals with portfolio management."""
        
        if not signals:
            self.logger.info("No signals to execute")
            return
        
        # Sort signals by confidence (highest first)
        sorted_signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)
        
        # Get current portfolio allocation
        if self.trading_controller:
            try:
                portfolio_status = self.trading_controller.get_portfolio_dashboard()
                current_positions = portfolio_status.get('portfolio_overview', {}).get('total_positions', 0)
                max_positions = self.config.risk_params['max_positions']
                
                self.logger.info(f"Current positions: {current_positions}, Max allowed: {max_positions}")
                
                # Calculate available position slots
                available_slots = max_positions - current_positions
                
                # Execute top signals within portfolio limits
                executed_count = 0
                for signal in sorted_signals:
                    if executed_count >= available_slots:
                        self.logger.info(f"Portfolio limit reached, skipping remaining signals")
                        break
                    
                    if signal['confidence'] >= 0.7:  # Only execute high-confidence signals
                        self._execute_trade_signal(signal)
                        executed_count += 1
                    else:
                        self.logger.info(f"Signal confidence too low for {signal['symbol']}: {signal['confidence']:.2f}")
                        
            except Exception as e:
                self.logger.error(f"Error executing multi-ticker trades: {str(e)}")
        else:
            self.logger.warning("No trading controller available for trade execution")
    
    def run_automated_trading_mode(self, duration_hours: int = 24):
        """Run comprehensive LSTM-based automated trading for all tickers."""
        
        self.logger.info(f"🤖 Starting LSTM-Based Automated Trading Mode for {duration_hours} hours")
        self.mode = "trading"
        self.running = True
        
        # Load all tickers from tickers.txt
        tickers = self._load_tickers_from_file()
        if not tickers:
            print("❌ No tickers found in tickers.txt")
            return
        
        print(f"🎯 Processing {len(tickers)} tickers: {', '.join(tickers)}")
        
        # Initialize components
        if not self._initialize_trading_components():
            return
            
        # Store training parameters
        self.max_epochs = getattr(self, 'max_epochs', 25)
        self.training_timeout = getattr(self, 'training_timeout', 300)
        
        # Initialize signal generator cache for efficient reuse
        self.signal_generator_cache = {}
        print(f"🧠 Initializing signal generator cache for {len(tickers)} tickers...")
            
        # Display header
        self._display_trading_header(tickers, duration_hours)
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        cycle_count = 0
        
        try:
            while self.running and datetime.now() < end_time:
                cycle_count += 1
                cycle_start = datetime.now()
                
                print(f"\n🔄 TRADING CYCLE #{cycle_count} - {cycle_start.strftime('%H:%M:%S')}")
                print("=" * 80)
                
                # Process each ticker
                all_signals = []
                for i, ticker in enumerate(tickers):
                    if not self.running:
                        break
                        
                    print(f"\n📊 Processing {ticker} ({i+1}/{len(tickers)})")
                    
                    # Step 1: Ensure LSTM model exists and is trained
                    model_status = self._ensure_lstm_model(ticker)
                    
                    # Step 2: Generate LSTM signal (optimized: reuses cached generator, only fetches data for this ticker)
                    signal_data = self._generate_lstm_signal_for_ticker(ticker)
                    
                    # Step 3: Process and enhance signal
                    if signal_data:
                        enhanced_signal = self._process_signal(signal_data)
                        if enhanced_signal:
                            all_signals.append(enhanced_signal)
                    
                    # Brief pause between tickers
                    time.sleep(1)
                
                # Step 4: Execute trades for all valid signals
                if all_signals:
                    self._execute_batch_trades(all_signals)
                else:
                    print("\n⚪ No trading signals generated this cycle")
                
                # Step 5: Portfolio status update
                self._display_portfolio_summary()
                
                # Step 6: Wait for next cycle (15 minutes)
                self._wait_for_next_cycle(900)  # 15 minutes
                
        except KeyboardInterrupt:
            self.logger.info("Automated trading interrupted by user")
        except Exception as e:
            self.logger.error(f"Automated trading error: {str(e)}")
            print(f"❌ Error: {str(e)}")
        finally:
            # Clean up resources
            if hasattr(self, 'signal_generator_cache'):
                self._clear_signal_generator_cache()
            self.running = False
            print("\n🛑 Automated trading stopped")
    
    def _load_tickers_from_file(self) -> List[str]:
        """Load ticker symbols from tickers.txt file."""
        try:
            tickers_file = os.path.join(os.path.dirname(__file__), 'tickers.txt')
            if not os.path.exists(tickers_file):
                self.logger.error("tickers.txt file not found")
                return []
            
            with open(tickers_file, 'r') as f:
                lines = f.readlines()
            
            # Filter out comments and empty lines
            tickers = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('//') and not line.startswith('#'):
                    tickers.append(line)
            
            return tickers
            
        except Exception as e:
            self.logger.error(f"Error loading tickers: {str(e)}")
            return []
    
    def _initialize_trading_components(self) -> bool:
        """Initialize all required trading components."""
        try:
            # Check trading controller
            if not self.trading_controller:
                print("❌ Trading controller not available. Check Alpaca credentials.")
                return False
            
            # Check data client
            if not self.data_client:
                print("❌ Data client not available.")
                return False
                
            # Check Alpaca client
            if not self.alpaca_client:
                print("❌ Alpaca client not available.")
                return False
            
            print("✅ All trading components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization error: {str(e)}")
            return False
    
    def _display_trading_header(self, tickers: List[str], duration_hours: int):
        """Display trading session header information."""
        print("\n" + "=" * 80)
        print("🚀 LSTM-BASED AUTOMATED CRYPTO TRADING")
        print("=" * 80)
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration: {duration_hours} hours")
        print(f"🎯 Tickers: {len(tickers)} symbols")
        print(f"🔄 Cycle Interval: 15 minutes")
        print(f"🧠 LSTM Training: {'✅ Enabled' if self.max_epochs > 0 else '❌ Disabled'}")
        print(f"⚡ Max Epochs: {self.max_epochs}")
        print(f"⏲️  Training Timeout: {self.training_timeout}s")
        print("=" * 80)
    
    def _ensure_lstm_model(self, ticker: str) -> Dict:
        """Ensure LSTM model exists and is trained for the ticker."""
        try:
            print(f"   🧠 Checking LSTM model for {ticker}...", end=" ")
            
            # Get cached signal generator (more efficient)
            signal_generator = self._get_or_create_signal_generator(ticker)
            
            # Check if model needs training
            model_path = signal_generator.get_model_path()
            needs_training = not os.path.exists(model_path)
            
            if needs_training:
                print("🔄 Training needed")
                print(f"   📚 Training LSTM model (max {self.max_epochs} epochs)...")
                
                # Train with timeout
                training_start = time.time()
                
                try:
                    # Start training in a separate thread to allow timeout
                    training_result = self._train_model_with_timeout(
                        signal_generator, 
                        self.training_timeout,
                        self.max_epochs
                    )
                    
                    training_time = time.time() - training_start
                    
                    if training_result.get('success', False):
                        print(f"   ✅ Model trained successfully in {training_time:.1f}s")
                        return {'status': 'trained', 'training_time': training_time}
                    else:
                        print(f"   ❌ Training failed: {training_result.get('error', 'Unknown error')}")
                        return {'status': 'failed', 'error': training_result.get('error')}
                        
                except Exception as e:
                    print(f"   ❌ Training error: {str(e)}")
                    return {'status': 'error', 'error': str(e)}
            else:
                print("✅ Model ready")
                return {'status': 'ready'}
                
        except Exception as e:
            print(f"   ❌ Model check error: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def _train_model_with_timeout(self, signal_generator, timeout: int, max_epochs: int) -> Dict:
        """Train LSTM model with timeout protection."""
        import signal as signal_module
        
        result = {'success': False, 'error': None}
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Training timeout after {timeout} seconds")
        
        # Set timeout
        signal_module.signal(signal_module.SIGALRM, timeout_handler)
        signal_module.alarm(timeout)
        
        try:
            # Train the model
            signal_generator.train_model(
                hours=self.config.lookback or 168,  # 1 week default
                epochs=max_epochs,
                verbose=False
            )
            result['success'] = True
            
        except TimeoutError as e:
            result['error'] = str(e)
        except Exception as e:
            result['error'] = f"Training failed: {str(e)}"
        finally:
            signal_module.alarm(0)  # Cancel timeout
        
        return result
    
    def _generate_lstm_signal_for_ticker(self, ticker: str) -> Optional[Dict]:
        """Generate LSTM trading signal for a specific ticker using cached generator."""
        try:
            print(f"   🎯 Generating signal...", end=" ")
            
            # Get cached signal generator (more efficient than creating new one)
            signal_generator = self._get_or_create_signal_generator(ticker)
            
            # Generate signal (this will fetch data only for this specific ticker)
            signal_result = signal_generator.generate_aggressive_signals(
                hours=self.config.lookback or 24,
                retrain_threshold=6
            )
            
            if 'error' in signal_result:
                print(f"❌ {signal_result['error']}")
                return None
            
            if signal_result and signal_result.get('signal') != 0:
                signal_type = 'BUY' if signal_result['signal'] > 0 else 'SELL'
                confidence = signal_result.get('confidence', 0)
                print(f"🎯 {signal_type} (confidence: {confidence:.2f})")
                
                return {
                    'symbol': ticker,
                    'signal': signal_type,
                    'confidence': confidence,
                    'current_price': signal_result.get('current_price', 0),
                    'predicted_price': signal_result.get('predicted_price'),
                    'timestamp': signal_result.get('timestamp', datetime.now().isoformat()),
                    'indicators': {
                        'momentum_score': signal_result.get('momentum_score', 0),
                        'volatility_breakout': signal_result.get('volatility_breakout', False),
                        'volume_surge': signal_result.get('volume_surge', 1.0),
                        'rsi_fast': signal_result.get('rsi_fast', 50),
                        'risk_multiplier': signal_result.get('risk_multiplier', 1.0)
                    },
                    'trade_recommendation': signal_result.get('trade_recommendation', {})
                }
            else:
                print("⚪ No signal")
                return None
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.logger.error(f"Signal generation error for {ticker}: {str(e)}")
            return None
    
    def _process_signal(self, signal_data: Dict) -> Optional[Dict]:
        """Process and enhance signal with additional validation."""
        try:
            # Basic validation
            confidence = signal_data.get('confidence', 0)
            if confidence < 0.6:  # Minimum confidence threshold
                print(f"   ⚠️  Low confidence ({confidence:.2f}), skipping")
                return None
            
            # Enhance with LLM if available
            if self.llm_client:
                enhanced_signal = self._enhance_signal_with_llm(signal_data)
                signal_data.update(enhanced_signal)
            
            # Add risk assessment
            signal_data['risk_assessment'] = self._assess_signal_risk(signal_data)
            
            print(f"   ✅ Signal processed (confidence: {confidence:.2f})")
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Signal processing error: {str(e)}")
            return None
    
    def _assess_signal_risk(self, signal_data: Dict) -> Dict:
        """Assess risk level of the trading signal."""
        try:
            confidence = signal_data.get('confidence', 0)
            indicators = signal_data.get('indicators', {})
            
            # Risk factors
            risk_score = 0
            
            # Confidence risk
            if confidence < 0.7:
                risk_score += 30
            elif confidence < 0.8:
                risk_score += 15
            
            # Volatility risk
            if indicators.get('volatility_breakout', False):
                risk_score += 20
            
            # RSI risk
            rsi = indicators.get('rsi_fast', 50)
            if rsi > 80 or rsi < 20:
                risk_score += 25
            
            # Volume risk
            volume_surge = indicators.get('volume_surge', 1.0)
            if volume_surge < 0.8:
                risk_score += 15
            
            # Risk level determination
            if risk_score <= 20:
                risk_level = 'LOW'
            elif risk_score <= 40:
                risk_level = 'MEDIUM'
            elif risk_score <= 60:
                risk_level = 'HIGH'
            else:
                risk_level = 'CRITICAL'
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': self._identify_risk_factors(risk_score, confidence, indicators)
            }
            
        except Exception as e:
            return {'risk_level': 'UNKNOWN', 'error': str(e)}
    
    def _identify_risk_factors(self, risk_score: int, confidence: float, indicators: Dict) -> List[str]:
        """Identify specific risk factors."""
        factors = []
        
        if confidence < 0.7:
            factors.append('Low prediction confidence')
        if indicators.get('volatility_breakout', False):
            factors.append('High volatility detected')
        if indicators.get('rsi_fast', 50) > 80:
            factors.append('Overbought conditions')
        elif indicators.get('rsi_fast', 50) < 20:
            factors.append('Oversold conditions')
        if indicators.get('volume_surge', 1.0) < 0.8:
            factors.append('Low trading volume')
        
        return factors
    
    def _execute_batch_trades(self, signals: List[Dict]):
        """Execute trades for all valid signals with portfolio management."""
        try:
            print(f"\n💰 EXECUTING TRADES ({len(signals)} signals)")
            print("-" * 50)
            
            total_executed = 0
            total_skipped = 0
            
            for signal in signals:
                try:
                    result = self._execute_enhanced_trade(signal)
                    if result.get('executed', False):
                        total_executed += 1
                    else:
                        total_skipped += 1
                        
                except Exception as e:
                    self.logger.error(f"Trade execution error for {signal.get('symbol')}: {str(e)}")
                    total_skipped += 1
            
            # Summary
            print(f"\n📊 EXECUTION SUMMARY")
            print(f"   ✅ Executed: {total_executed}")
            print(f"   ⚠️  Skipped: {total_skipped}")
            print(f"   📈 Success Rate: {(total_executed/(total_executed+total_skipped)*100) if (total_executed+total_skipped) > 0 else 0:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Batch trade execution error: {str(e)}")
    
    def _execute_enhanced_trade(self, signal_data: Dict) -> Dict:
        """Execute trade with enhanced validation and risk management."""
        try:
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            risk_assessment = signal_data.get('risk_assessment', {})
            
            print(f"🎯 {symbol}: {signal} (confidence: {confidence:.2f})", end=" ")
            
            # Risk-based filtering
            risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
            if risk_level == 'CRITICAL':
                print("❌ CRITICAL RISK - Trade blocked")
                return {'executed': False, 'reason': 'Critical risk level'}
            
            # Confidence filtering
            min_confidence = 0.75 if risk_level == 'HIGH' else 0.65
            if confidence < min_confidence:
                print(f"❌ Low confidence for {risk_level} risk")
                return {'executed': False, 'reason': 'Insufficient confidence'}
            
            # Calculate position size based on risk
            position_size = self._calculate_risk_adjusted_position_size(signal_data)
            
            if not self.alpaca_client:
                print("❌ No trading client")
                return {'executed': False, 'reason': 'No trading client'}
            
            # Execute the trade
            alpaca_symbol = symbol.replace('/', '')
            
            try:
                if signal == 'BUY':
                    result = self.alpaca_client.place_order(
                        symbol=alpaca_symbol,
                        qty=position_size,
                        side='buy',
                        type='market'
                    )
                elif signal == 'SELL':
                    result = self.alpaca_client.place_order(
                        symbol=alpaca_symbol,
                        qty=position_size,
                        side='sell',
                        type='market'
                    )
                else:
                    print("❌ Invalid signal")
                    return {'executed': False, 'reason': 'Invalid signal'}
                
                if result and 'id' in result:
                    print(f"✅ ORDER PLACED (${position_size:.2f})")
                    self.logger.info(f"Trade executed: {symbol} {signal} ${position_size:.2f}")
                    return {'executed': True, 'order_id': result['id'], 'amount': position_size}
                else:
                    print("❌ Order failed")
                    return {'executed': False, 'reason': 'Order placement failed'}
                
            except Exception as e:
                print(f"❌ Execution error: {str(e)}")
                return {'executed': False, 'reason': str(e)}
                
        except Exception as e:
            self.logger.error(f"Enhanced trade execution error: {str(e)}")
            return {'executed': False, 'reason': str(e)}
    
    def _calculate_risk_adjusted_position_size(self, signal_data: Dict) -> float:
        """Calculate position size based on risk assessment."""
        try:
            base_position_size = self.config.risk_params.get('position_size', 100.0)
            risk_level = signal_data.get('risk_assessment', {}).get('risk_level', 'MEDIUM')
            confidence = signal_data.get('confidence', 0.5)
            
            # Risk multipliers
            risk_multipliers = {
                'LOW': 1.2,
                'MEDIUM': 1.0,
                'HIGH': 0.7,
                'CRITICAL': 0.3
            }
            
            # Confidence multiplier
            confidence_multiplier = min(confidence * 1.5, 1.2)
            
            # Calculate final size
            risk_multiplier = risk_multipliers.get(risk_level, 0.5)
            final_size = base_position_size * risk_multiplier * confidence_multiplier
            
            # Ensure minimum and maximum bounds
            min_size = 10.0
            max_size = base_position_size * 2.0
            
            return max(min_size, min(final_size, max_size))
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {str(e)}")
            return self.config.risk_params.get('position_size', 100.0) * 0.5
    
    def _display_portfolio_summary(self):
        """Display current portfolio summary."""
        try:
            if not self.alpaca_client:
                return
                
            print(f"\n💼 PORTFOLIO STATUS")
            print("-" * 30)
            
            # Get account info
            account = self.alpaca_client.get_account()
            if account:
                equity = float(account.get('equity', 0))
                buying_power = float(account.get('buying_power', 0))
                pnl = float(account.get('unrealized_pnl', 0))
                
                print(f"💰 Equity: {format_currency(equity)}")
                print(f"💳 Buying Power: {format_currency(buying_power)}")
                print(f"📊 Unrealized P&L: {format_currency(pnl)}")
            
            # Get positions
            positions = self.alpaca_client.list_positions()
            if positions:
                print(f"📈 Active Positions: {len(positions)}")
                for pos in positions[:3]:  # Show top 3
                    symbol = pos.get('symbol', 'Unknown')
                    qty = float(pos.get('qty', 0))
                    market_value = float(pos.get('market_value', 0))
                    unrealized_pnl = float(pos.get('unrealized_pnl', 0))
                    
                    print(f"   {symbol}: {qty:.4f} units, {format_currency(market_value)} ({format_currency(unrealized_pnl)})")
            else:
                print("📈 No active positions")
                
        except Exception as e:
            self.logger.error(f"Portfolio summary error: {str(e)}")
            print("❌ Portfolio data unavailable")
    
    def _wait_for_next_cycle(self, wait_seconds: int):
        """Wait for next trading cycle with progress indication."""
        try:
            print(f"\n⏱️  Waiting {wait_seconds//60} minutes until next cycle...")
            
            # Show progress every 30 seconds
            for i in range(0, wait_seconds, 30):
                if not self.running:
                    break
                    
                remaining = wait_seconds - i
                progress = (i / wait_seconds) * 100
                
                print(f"\r   {create_progress_bar(progress, 40)} {remaining//60:02d}:{remaining%60:02d} remaining", end="")
                
                # Sleep for 30 seconds or until stopped
                for j in range(30):
                    if not self.running:
                        break
                    time.sleep(1)
            
            print()  # New line after progress
            
        except KeyboardInterrupt:
            self.running = False
    
    def _get_or_create_signal_generator(self, ticker: str):
        """Get cached signal generator or create new one for ticker."""
        if ticker not in self.signal_generator_cache:
            self.logger.debug(f"Creating new signal generator for {ticker}")
            self.signal_generator_cache[ticker] = AggressiveCryptoSignalGenerator(ticker=ticker)
        return self.signal_generator_cache[ticker]

    def _clear_signal_generator_cache(self):
        """Clear signal generator cache to free memory."""
        self.signal_generator_cache.clear()
        self.logger.debug("Signal generator cache cleared")


def main():
    """Main entry point with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot - Comprehensive Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_app.py --mode dashboard                         # Interactive dashboard
  python main_app.py --mode trading --duration 24            # Automated trading for 24 hours
  python main_app.py --mode lstm-trading --duration 24       # LSTM-based trading for all tickers
  python main_app.py --mode lstm-trading --max-epochs 50     # LSTM trading with custom epochs
  python main_app.py --mode analysis --period 30             # 30-day performance analysis
  python main_app.py --mode backtest --hours 720             # Backtest for 720 hours
  python main_app.py --mode data --duration 1                # Data collection for 1 hour
  python main_app.py --mode setup                            # System setup and validation
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['dashboard', 'trading', 'lstm-trading', 'analysis', 'backtest', 'data', 'setup'],
        default='dashboard',
        help='Execution mode (default: dashboard)'
    )
    
    parser.add_argument(
        '--ticker',
        default='BTC',
        help='Crypto ticker symbol (default: BTC)'
    )
    
    parser.add_argument(
        '--multi-ticker',
        action='store_true',
        help='Enable multi-ticker trading mode'
    )
    
    parser.add_argument(
        '--max-tickers',
        type=int,
        default=3,
        help='Maximum number of active tickers in multi-ticker mode (default: 3)'
    )
    
    parser.add_argument(
        '--ticker-allocation',
        type=float,
        default=0.33,
        help='Portfolio allocation per ticker in multi-ticker mode (default: 0.33)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=24,
        help='Duration in hours for trading/data modes (default: 24)'
    )
    
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=25,
        help='Maximum training epochs for LSTM model (default: 25)'
    )
    
    parser.add_argument(
        '--training-timeout',
        type=int,
        default=300,  # 5 minutes
        help='Maximum training time in seconds (default: 300)'
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
            
        # Configure multi-ticker settings if enabled
        if args.multi_ticker:
            config.multi_ticker_enabled = True
            config.max_active_tickers = args.max_tickers
            config.ticker_allocation = args.ticker_allocation
            print(f"🔀 Multi-ticker mode enabled")
            print(f"📊 Max tickers: {args.max_tickers}")
            print(f"💰 Per-ticker allocation: {args.ticker_allocation:.2%}")
        
        # Initialize application
        app = CryptoTradingBotApp(config)
        
        # Set training parameters
        app.max_epochs = args.max_epochs
        app.training_timeout = args.training_timeout
        
        print(f"🚀 Starting Crypto Trading Bot in {args.mode.upper()} mode")
        if args.multi_ticker:
            active_tickers = config.get_active_tickers()
            print(f"📊 Active tickers: {', '.join(active_tickers[:args.max_tickers])}")
        else:
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
            
        elif args.mode == 'lstm-trading':
            app.run_automated_trading_mode(duration_hours=args.duration)
            
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
